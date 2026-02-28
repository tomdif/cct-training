"""
Retrieval CCT eval: tests whether retrieving past steps' real KV caches
improves PPL over pseudo-tokens alone.

Runs FIVE conditions with per-step PPL breakdown:
  1. Pseudo-only     — existing pseudo-token delivery (reference)
  2. Retrieval-only  — past KV from K most recent steps, no pseudo-tokens
  3. Hybrid          — pseudo-tokens + retrieved KV caches
  4. Isolated        — each step in isolation (reference)
  5. Sliding window  — plain 800-token prior context, no CCT (prior art baseline)

The key insight: summaries carry ~75% topic signal but miss specific facts.
Retrieved KV caches provide full-fidelity past context. The hybrid combines
cheap topic context (pseudo-tokens) with specific recall (retrieved KV).

Usage:
  python scripts/eval_retrieval.py \
    --config configs/tier3_410m_pseudo_tokens_O.yaml \
    --cct-checkpoint ./checkpoints-run-O-contrastive/cct-6000.pt \
    --eval-batches 100 --seq-len 2048 \
    --retrieval-k 2 --retrieval-strategy recent
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_pipeline import (
    extend_tokenizer,
    annotate_step_boundaries,
    get_step_boundary_positions,
    get_step_ranges,
)
from src.model.commitment_head import CommitmentHead
from src.model.summary_buffer import SummaryBuffer
from src.model.pseudo_token_decoder import PseudoTokenDecoder


# ══════════════════════════════════════════════════════════
# Streaming eval dataset (same as eval_per_step.py)
# ══════════════════════════════════════════════════════════

class StreamingEvalDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, split, tokenizer, seq_len, step_token_id,
                 step_length, skip_examples=50000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.step_token_id = step_token_id
        self.step_length = step_length
        self.skip_examples = skip_examples
        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def __iter__(self):
        buffer = []
        skipped = 0
        for example in self.dataset:
            if skipped < self.skip_examples:
                skipped += 1
                continue
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                annotated = annotate_step_boundaries(
                    chunk, self.step_token_id, "fixed", self.step_length
                )
                annotated = annotated[:self.seq_len]
                yield torch.tensor(annotated, dtype=torch.long)


# ══════════════════════════════════════════════════════════
# KV Cache Utilities
# ══════════════════════════════════════════════════════════

def _cache_to_tuples(cache, n_layers):
    """Convert any cache format to list of (key, value) tuples."""
    result = []
    # Newer transformers: .layers list with .keys/.values per layer
    if hasattr(cache, 'layers') and isinstance(cache.layers, list):
        for layer in cache.layers:
            result.append((layer.keys, layer.values))
        return result
    # Mid-version: .key_cache/.value_cache lists
    if hasattr(cache, 'key_cache'):
        for i in range(n_layers):
            result.append((cache.key_cache[i], cache.value_cache[i]))
        return result
    # Legacy: tuple of (key, value) per layer
    try:
        for i in range(n_layers):
            item = cache[i]
            result.append((item[0], item[1]))
        return result
    except (TypeError, IndexError, KeyError):
        pass
    raise ValueError(
        f"Cannot extract KV from {type(cache)}. "
        f"Attrs: {[a for a in dir(cache) if not a.startswith('__')]}"
    )


def _tuples_to_cache(kv_tuples):
    """Convert list/tuple of (key, value) back to DynamicCache for model input."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_tuples):
        cache.update(k, v, layer_idx)
    return cache


def extract_step_kv(output_cache, step_token_len, n_layers):
    """Extract KV for only the step's own tokens from model output cache.

    Returns a tuple of (key, value) per layer (plain tensors, no DynamicCache).
    """
    raw = _cache_to_tuples(output_cache, n_layers)
    result = []
    for k_full, v_full in raw:
        k_step = k_full[:, :, -step_token_len:, :].clone()
        v_step = v_full[:, :, -step_token_len:, :].clone()
        result.append((k_step, v_step))
    return tuple(result)


def merge_kv_caches(caches, n_layers):
    """Concatenate multiple step KV caches along the sequence dimension.

    Input: list of tuple-of-(key,value) caches.
    Returns: tuple of (key, value) per layer.
    """
    if not caches:
        return None
    if len(caches) == 1:
        return caches[0]

    result = []
    for layer_idx in range(n_layers):
        keys = [c[layer_idx][0] for c in caches]
        values = [c[layer_idx][1] for c in caches]
        result.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    return tuple(result)


def estimate_kv_bank_memory(kv_bank, n_layers):
    """Estimate GPU memory in MB used by all KV bank entries."""
    total_bytes = 0
    for cache in kv_bank:
        for layer_idx in range(n_layers):
            k, v = cache[layer_idx]
            total_bytes += k.nelement() * k.element_size()
            total_bytes += v.nelement() * v.element_size()
    return total_bytes / (1024 * 1024)


def select_retrieval_indices(strategy, kv_bank_size, retrieval_k,
                             bank_summaries=None, current_summary=None):
    """Select which past step KV caches to retrieve.

    Returns list of bank indices in chronological order.
    """
    if kv_bank_size == 0:
        return []

    if strategy == "recent":
        start = max(0, kv_bank_size - retrieval_k)
        return list(range(start, kv_bank_size))

    elif strategy == "all":
        return list(range(kv_bank_size))

    elif strategy == "similarity":
        if bank_summaries is None or current_summary is None or len(bank_summaries) == 0:
            # Fall back to recent
            start = max(0, kv_bank_size - retrieval_k)
            return list(range(start, kv_bank_size))

        # Cosine similarity: current summary vs each stored summary
        # Use batch element 0 (summaries are uniform across batch for same data)
        curr = current_summary[0]  # (d_summary,)
        sims = []
        for i in range(min(kv_bank_size, len(bank_summaries))):
            s = bank_summaries[i][0]  # (d_summary,)
            cos_sim = F.cosine_similarity(
                curr.unsqueeze(0), s.unsqueeze(0)
            ).item()
            sims.append((i, cos_sim))

        # Top-K by similarity, returned in chronological order
        sims.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([idx for idx, _ in sims[:retrieval_k]])
        return selected

    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")


# ══════════════════════════════════════════════════════════
# Sliding window baseline (no CCT, pure prior art)
# ══════════════════════════════════════════════════════════

def compute_sliding_window(model, dataloader, step_token_id, device,
                           max_batches, window_tokens=800):
    """Sliding window baseline: each step sees window_tokens of raw prior context.

    No commitment head, no pseudo-tokens, no GRU — just the frozen model
    with a limited context window. This is the prior-art comparison.
    """
    model.eval()
    step_loss = defaultdict(float)
    step_tokens_dict = defaultdict(int)
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(seq_len, boundary_positions)

            for step_idx, (start, end) in enumerate(step_ranges):
                step_len = end - start
                if step_len <= 0:
                    continue

                # Window: up to window_tokens before current step + current step
                win_start = max(0, start - window_tokens)
                win_ids = input_ids[:, win_start:end]

                # Absolute position IDs
                win_positions = torch.arange(
                    win_start, end, device=device
                ).unsqueeze(0).expand(batch_size, -1)

                # Full attention within window (no CCT, no past_key_values)
                outputs = model(
                    input_ids=win_ids,
                    position_ids=win_positions,
                )

                # Loss on current step tokens only
                offset = start - win_start
                step_logits = outputs.logits[:, offset:offset + step_len - 1, :]
                step_labels = input_ids[:, start + 1:end].clone()
                step_labels[step_labels == step_token_id] = -100

                # Cross-boundary: logit at offset-1 predicts first step token
                if offset > 0:
                    cross_logit = outputs.logits[:, offset - 1:offset, :]
                    first_label = input_ids[:, start:start + 1].clone()
                    first_label[first_label == step_token_id] = -100
                    step_logits = torch.cat([cross_logit, step_logits], dim=1)
                    step_labels = torch.cat([first_label, step_labels], dim=1)

                loss = F.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    step_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (step_labels.reshape(-1) != -100).sum().item()
                step_loss[step_idx] += loss.item()
                step_tokens_dict[step_idx] += n_valid

            n_batches += 1
            if n_batches % 50 == 0:
                total_l = sum(step_loss.values())
                total_t = sum(step_tokens_dict.values())
                avg = total_l / total_t if total_t > 0 else 0
                print(f"  batch {n_batches}/{max_batches} | "
                      f"avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    return step_loss, step_tokens_dict, n_batches


# ══════════════════════════════════════════════════════════
# Core eval function (handles all 4 conditions)
# ══════════════════════════════════════════════════════════

def compute_retrieval_condition(
    model, commitment_head, pseudo_decoder, embed_layer,
    dataloader, step_token_id, device, max_batches,
    n_summary_tokens_K=1,
    use_pseudo_tokens=True,
    use_kv_retrieval=True,
    retrieval_k=2,
    retrieval_strategy="recent",
    max_bank_size=8,
    config=None,
):
    """Run one eval condition: process sequences step by step.

    Conditions controlled by flags:
      pseudo_only:    use_pseudo_tokens=True,  use_kv_retrieval=False
      retrieval_only: use_pseudo_tokens=False, use_kv_retrieval=True
      hybrid:         use_pseudo_tokens=True,  use_kv_retrieval=True
      isolated:       use_pseudo_tokens=False, use_kv_retrieval=False
    """
    model.eval()
    commitment_head.eval()
    if pseudo_decoder is not None:
        pseudo_decoder.eval()

    step_loss = defaultdict(float)
    step_tokens = defaultdict(int)
    n_batches = 0
    n_layers = model.config.num_hidden_layers
    peak_kv_mb = 0.0
    K = n_summary_tokens_K

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(seq_len, boundary_positions)

            committed_summaries = []  # Full list (for GRU recurrence)
            kv_bank = []              # Stored KV caches (max_bank_size)
            bank_summaries = []       # Summaries aligned with kv_bank

            for step_idx, (start, end) in enumerate(step_ranges):
                step_len = end - start
                if step_len <= 0:
                    continue

                step_token_ids = input_ids[:, start:end]
                step_embeds = embed_layer(step_token_ids)

                step_positions = torch.arange(
                    start, end, device=device
                ).unsqueeze(0).expand(batch_size, -1)

                n_prior_steps = len(committed_summaries)
                n_prepend = 0
                retrieved_kv = None

                # ---- Pseudo-token prefix ----
                if (use_pseudo_tokens and pseudo_decoder is not None
                        and n_prior_steps > 0):
                    P = pseudo_decoder.n_pseudo_tokens
                    n_prepend = n_prior_steps * P
                    summary_embeds_list = []
                    summary_pos_list = []

                    for s_idx in range(n_prior_steps):
                        s = committed_summaries[s_idx]
                        if K > 1:
                            s = s[:, 0, :]
                        decoded = pseudo_decoder(s).to(step_embeds.dtype)
                        summary_embeds_list.append(decoded)

                    for i in range(n_prepend):
                        summary_pos_list.append(max(0, start - n_prepend + i))

                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)
                    summary_positions = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([summary_positions, step_positions], dim=1)
                else:
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                # ---- Retrieved KV cache ----
                if use_kv_retrieval and len(kv_bank) > 0:
                    indices = select_retrieval_indices(
                        retrieval_strategy,
                        len(kv_bank),
                        retrieval_k,
                        bank_summaries=bank_summaries,
                        current_summary=(
                            committed_summaries[-1] if committed_summaries else None
                        ),
                    )
                    if indices:
                        selected_caches = [kv_bank[i] for i in indices]
                        retrieved_kv = merge_kv_caches(selected_caches, n_layers)

                # ---- Forward pass ----
                past_kv_arg = (
                    _tuples_to_cache(retrieved_kv)
                    if retrieved_kv is not None else None
                )
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    past_key_values=past_kv_arg,
                    use_cache=True,
                )

                # ---- Store this step's KV in bank ----
                is_last_step = (step_idx == len(step_ranges) - 1)
                if not is_last_step and outputs.past_key_values is not None:
                    step_kv = extract_step_kv(
                        outputs.past_key_values, step_len, n_layers
                    )
                    kv_bank.append(step_kv)

                    # Evict oldest if over budget
                    while len(kv_bank) > max_bank_size:
                        evicted = kv_bank.pop(0)
                        del evicted
                        if bank_summaries:
                            bank_summaries.pop(0)

                    # Track peak memory
                    bank_mb = estimate_kv_bank_memory(kv_bank, n_layers)
                    peak_kv_mb = max(peak_kv_mb, bank_mb)

                # ---- Commitment summary (GRU recurrence) ----
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][:, n_prepend:, :]
                    prev_sum = (
                        committed_summaries[-1] if committed_summaries else None
                    )
                    summary = commitment_head(
                        step_hidden.float(), prev_summary=prev_sum
                    )
                    committed_summaries.append(summary)
                    bank_summaries.append(summary)

                # ---- Loss computation ----
                step_logits = outputs.logits[
                    :, n_prepend:n_prepend + step_len - 1, :
                ]
                step_labels = step_token_ids[:, 1:].clone()
                step_labels[step_labels == step_token_id] = -100

                # Cross-boundary from last pseudo-token
                if n_prepend > 0:
                    cross_logit = outputs.logits[
                        :, n_prepend - 1:n_prepend, :
                    ]
                    first_label = step_token_ids[:, 0:1].clone()
                    first_label[first_label == step_token_id] = -100
                    step_logits = torch.cat([cross_logit, step_logits], dim=1)
                    step_labels = torch.cat([first_label, step_labels], dim=1)

                loss = F.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    step_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (step_labels.reshape(-1) != -100).sum().item()
                step_loss[step_idx] += loss.item()
                step_tokens[step_idx] += n_valid

            # Clear bank between sequences
            kv_bank.clear()
            bank_summaries.clear()

            n_batches += 1
            if n_batches % 50 == 0:
                total_l = sum(step_loss.values())
                total_t = sum(step_tokens.values())
                avg = total_l / total_t if total_t > 0 else 0
                print(f"  batch {n_batches}/{max_batches} | "
                      f"avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    return step_loss, step_tokens, n_batches, peak_kv_mb


# ══════════════════════════════════════════════════════════
# Output formatting
# ══════════════════════════════════════════════════════════

def format_retrieval_table(results_dict, max_step):
    """Format per-step PPL comparison across all conditions."""
    has_sw = "sliding_window" in results_dict
    w = 130 if has_sw else 110
    lines = []
    lines.append("")
    lines.append("=" * w)
    lines.append("RETRIEVAL CCT: PER-STEP PERPLEXITY COMPARISON")
    lines.append("=" * w)

    hdr1 = (
        f"  {'Step':>4}  {'Pseudo':>10}  {'Retrieval':>10}  "
        f"{'Hybrid':>10}  {'Isolated':>10}"
    )
    hdr2 = (
        f"  {'':>4}  {'Only':>10}  {'Only':>10}  "
        f"{'P+R':>10}  {'(no ctx)':>10}"
    )
    if has_sw:
        hdr1 += f"  {'SlidWin':>10}"
        hdr2 += f"  {'(prior)':>10}"
    hdr1 += f"  {'Hyb-Iso':>10}  {'Hyb-Psd':>10}  {'Hyb-SW':>10}"
    hdr2 += f"  {'benefit':>10}  {'benefit':>10}  {'vs prior':>10}"
    lines.append(hdr1)
    lines.append(hdr2)
    lines.append("-" * w)

    def _row(label, ppls):
        p = ppls.get("pseudo_only", float('nan'))
        r = ppls.get("retrieval_only", float('nan'))
        h = ppls.get("hybrid", float('nan'))
        iso = ppls.get("isolated", float('nan'))
        sw = ppls.get("sliding_window", float('nan'))

        hyb_iso = iso - h if not (math.isnan(iso) or math.isnan(h)) else float('nan')
        hyb_psd = p - h if not (math.isnan(p) or math.isnan(h)) else float('nan')
        hyb_sw = sw - h if not (math.isnan(sw) or math.isnan(h)) else float('nan')

        row = (
            f"  {label:>4}  {p:>10.2f}  {r:>10.2f}  "
            f"{h:>10.2f}  {iso:>10.2f}"
        )
        if has_sw:
            row += f"  {sw:>10.2f}"
        row += f"  {hyb_iso:>+10.2f}  {hyb_psd:>+10.2f}  {hyb_sw:>+10.2f}"
        return row

    for step_idx in range(max_step + 1):
        ppls = {}
        for cond_name, (s_loss, s_tok) in results_dict.items():
            l = s_loss.get(step_idx, 0)
            t = s_tok.get(step_idx, 0)
            ppls[cond_name] = math.exp(l / t) if t > 0 else float('nan')
        lines.append(_row(str(step_idx), ppls))

    # Totals
    lines.append("-" * w)
    totals = {}
    for cond_name, (s_loss, s_tok) in results_dict.items():
        tl = sum(s_loss.values())
        tt = sum(s_tok.values())
        totals[cond_name] = math.exp(tl / tt) if tt > 0 else float('nan')
    lines.append(_row("ALL", totals))

    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  Hyb-Iso:  positive = hybrid CCT helps over isolation (total benefit)")
    lines.append("  Hyb-Psd:  positive = adding KV retrieval to pseudo-tokens helps")
    lines.append("  Hyb-SW:   positive = hybrid CCT beats plain sliding window (CRITICAL)")
    lines.append("")
    lines.append("  If Hyb-SW > 0: CCT adds genuine value over prior-art sliding window")
    lines.append("  If Hyb-SW ~ 0: retrieved KV = sliding window, pseudo-tokens add nothing")
    lines.append("  If Hyb-SW < 0: plain sliding window beats CCT (bad)")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Retrieval CCT eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: from config)")
    parser.add_argument("--retrieval-k", type=int, default=2,
                        help="Number of past steps to retrieve")
    parser.add_argument("--retrieval-strategy", type=str, default="recent",
                        choices=["recent", "all", "similarity"],
                        help="How to select which past steps to retrieve")
    parser.add_argument("--max-bank-size", type=int, default=8,
                        help="Max KV caches stored (evicts oldest)")
    parser.add_argument("--window-tokens", type=int, default=800,
                        help="Sliding window size in tokens (prior art baseline)")
    parser.add_argument("--skip-pseudo-only", action="store_true")
    parser.add_argument("--skip-isolated", action="store_true")
    parser.add_argument("--skip-sliding-window", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seq_len = args.seq_len or config["seq_len"]
    step_length = config["step_length"]
    n_summary_tokens = config.get("n_summary_tokens", 1)

    print(f"Config: seq_len={seq_len}, step_length={step_length}, "
          f"n_summary_tokens={n_summary_tokens}")
    print(f"Expected steps per sequence: ~{seq_len // (step_length + 1)}")
    print(f"Retrieval: strategy={args.retrieval_strategy}, "
          f"k={args.retrieval_k}, max_bank={args.max_bank_size}")
    print(f"Sliding window: {args.window_tokens} tokens (prior art baseline)")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found")

    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_token_id = extend_tokenizer(tokenizer)

    batch_size = config.get("batch_size", 2)

    # ---- Load model ----
    print("\nLoading model...")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn_impl = config.get("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=model_dtype,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)

    # Apply LoRA if checkpoint used it
    if config.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        from src.model.model_utils import detect_lora_target_modules
        target_modules = config.get("lora_target_modules", [])
        if not target_modules:
            target_modules = detect_lora_target_modules(model)
            print(f"  LoRA auto-detected target modules: {target_modules}")
        lora_config = LoraConfig(
            r=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=0.0,
            target_modules=target_modules,
            layers_to_transform=list(range(
                config.get("lora_layers_min", 12),
                config.get("lora_layers_max", 23) + 1,
            )),
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    cct_state = {
        k: v.to(model_dtype) if v.is_floating_point() else v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(cct_state)

    # ---- Load CCT components ----
    commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config["d_summary"],
        d_bottleneck=config["d_bottleneck"],
        use_tanh=config.get("use_tanh", True),
        use_l2_norm=config.get("use_l2_norm", True),
        noise_injection=config.get("noise_injection", False),
        n_summary_tokens=config.get("n_summary_tokens", 1),
        recurrent=config.get("recurrent_commitment", False),
    ).to(device)
    commitment_head.load_state_dict(ckpt["commitment_head_state_dict"])

    # Load pseudo-token decoder
    pseudo_decoder = None
    if ckpt.get("pseudo_decoder_state_dict") is not None:
        pseudo_decoder = PseudoTokenDecoder(
            d_summary=config["d_summary"],
            d_model=config["d_model"],
            n_pseudo_tokens=config.get("n_pseudo_tokens", 8),
            hidden_dim=config.get("pseudo_decoder_hidden", 512),
            device=device,
        )
        pseudo_decoder.load_state_dict(ckpt["pseudo_decoder_state_dict"])
        pseudo_decoder.eval()
        print(f"  Loaded pseudo-token decoder: {pseudo_decoder.param_count():,} params "
              f"({config.get('n_pseudo_tokens', 8)} tokens)")

    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.gpt_neox.embed_in

    n_layers = model.config.num_hidden_layers
    print(f"  Model: {model_name} ({n_layers} layers)")

    # Estimate per-step KV memory
    n_heads = model.config.num_attention_heads
    d_head = config["d_model"] // n_heads
    per_step_kv_mb = (
        n_layers * 2 * n_heads * step_length * d_head * 2  # bf16
    ) / (1024 * 1024)
    print(f"  Per-step KV: ~{per_step_kv_mb:.0f} MB, "
          f"bank of {args.max_bank_size}: ~{per_step_kv_mb * args.max_bank_size:.0f} MB")

    def make_loader():
        ds = StreamingEvalDataset(
            dataset_name=config.get("validation_dataset", config["dataset"]),
            split=config.get("validation_split", "train"),
            tokenizer=tokenizer, seq_len=seq_len,
            step_token_id=step_token_id, step_length=step_length,
            skip_examples=50000,
        )
        return DataLoader(ds, batch_size=batch_size)

    results_dict = {}
    memory_stats = {}

    # ── Condition 1: Pseudo-only ──
    if not args.skip_pseudo_only:
        print("\n" + "=" * 60)
        print("CONDITION 1: PSEUDO-TOKEN ONLY (reference)")
        print("=" * 60)
        ps_loss, ps_tok, ps_n, ps_mem = compute_retrieval_condition(
            model, commitment_head, pseudo_decoder, embed_layer,
            make_loader(), step_token_id, device, args.eval_batches,
            n_summary_tokens_K=n_summary_tokens,
            use_pseudo_tokens=True, use_kv_retrieval=False,
            config=config,
        )
        total = sum(ps_loss.values()) / max(sum(ps_tok.values()), 1)
        print(f"\n  Pseudo-only total: loss={total:.4f} ppl={math.exp(total):.2f}")
        for si in sorted(ps_loss.keys()):
            if ps_tok[si] > 0:
                print(f"    step {si}: ppl={math.exp(ps_loss[si] / ps_tok[si]):.2f} "
                      f"({ps_tok[si]} tokens)")
        results_dict["pseudo_only"] = (ps_loss, ps_tok)
        memory_stats["pseudo_only_peak_kv_mb"] = ps_mem

    # ── Condition 2: Retrieval-only ──
    print("\n" + "=" * 60)
    print(f"CONDITION 2: RETRIEVAL ONLY ({args.retrieval_strategy}-{args.retrieval_k})")
    print("=" * 60)
    ret_loss, ret_tok, ret_n, ret_mem = compute_retrieval_condition(
        model, commitment_head, pseudo_decoder, embed_layer,
        make_loader(), step_token_id, device, args.eval_batches,
        n_summary_tokens_K=n_summary_tokens,
        use_pseudo_tokens=False, use_kv_retrieval=True,
        retrieval_k=args.retrieval_k,
        retrieval_strategy=args.retrieval_strategy,
        max_bank_size=args.max_bank_size,
        config=config,
    )
    total = sum(ret_loss.values()) / max(sum(ret_tok.values()), 1)
    print(f"\n  Retrieval-only total: loss={total:.4f} ppl={math.exp(total):.2f}")
    for si in sorted(ret_loss.keys()):
        if ret_tok[si] > 0:
            print(f"    step {si}: ppl={math.exp(ret_loss[si] / ret_tok[si]):.2f} "
                  f"({ret_tok[si]} tokens)")
    results_dict["retrieval_only"] = (ret_loss, ret_tok)
    memory_stats["retrieval_only_peak_kv_mb"] = ret_mem
    print(f"  Peak KV bank memory: {ret_mem:.0f} MB")

    # ── Condition 3: Hybrid ──
    print("\n" + "=" * 60)
    print(f"CONDITION 3: HYBRID (pseudo + {args.retrieval_strategy}-{args.retrieval_k})")
    print("=" * 60)
    hyb_loss, hyb_tok, hyb_n, hyb_mem = compute_retrieval_condition(
        model, commitment_head, pseudo_decoder, embed_layer,
        make_loader(), step_token_id, device, args.eval_batches,
        n_summary_tokens_K=n_summary_tokens,
        use_pseudo_tokens=True, use_kv_retrieval=True,
        retrieval_k=args.retrieval_k,
        retrieval_strategy=args.retrieval_strategy,
        max_bank_size=args.max_bank_size,
        config=config,
    )
    total = sum(hyb_loss.values()) / max(sum(hyb_tok.values()), 1)
    print(f"\n  Hybrid total: loss={total:.4f} ppl={math.exp(total):.2f}")
    for si in sorted(hyb_loss.keys()):
        if hyb_tok[si] > 0:
            print(f"    step {si}: ppl={math.exp(hyb_loss[si] / hyb_tok[si]):.2f} "
                  f"({hyb_tok[si]} tokens)")
    results_dict["hybrid"] = (hyb_loss, hyb_tok)
    memory_stats["hybrid_peak_kv_mb"] = hyb_mem
    print(f"  Peak KV bank memory: {hyb_mem:.0f} MB")

    # ── Condition 4: Isolated ──
    if not args.skip_isolated:
        print("\n" + "=" * 60)
        print("CONDITION 4: ISOLATED (no cross-step context)")
        print("=" * 60)
        iso_loss, iso_tok, iso_n, iso_mem = compute_retrieval_condition(
            model, commitment_head, pseudo_decoder, embed_layer,
            make_loader(), step_token_id, device, args.eval_batches,
            n_summary_tokens_K=n_summary_tokens,
            use_pseudo_tokens=False, use_kv_retrieval=False,
            config=config,
        )
        total = sum(iso_loss.values()) / max(sum(iso_tok.values()), 1)
        print(f"\n  Isolated total: loss={total:.4f} ppl={math.exp(total):.2f}")
        for si in sorted(iso_loss.keys()):
            if iso_tok[si] > 0:
                print(f"    step {si}: ppl={math.exp(iso_loss[si] / iso_tok[si]):.2f} "
                      f"({iso_tok[si]} tokens)")
        results_dict["isolated"] = (iso_loss, iso_tok)

    # ── Condition 5: Sliding window (prior art baseline) ──
    if not args.skip_sliding_window:
        print("\n" + "=" * 60)
        print(f"CONDITION 5: SLIDING WINDOW ({args.window_tokens} tokens, no CCT)")
        print("=" * 60)
        sw_loss, sw_tok, sw_n = compute_sliding_window(
            model, make_loader(), step_token_id, device,
            args.eval_batches, window_tokens=args.window_tokens,
        )
        total = sum(sw_loss.values()) / max(sum(sw_tok.values()), 1)
        print(f"\n  Sliding window total: loss={total:.4f} ppl={math.exp(total):.2f}")
        for si in sorted(sw_loss.keys()):
            if sw_tok[si] > 0:
                print(f"    step {si}: ppl={math.exp(sw_loss[si] / sw_tok[si]):.2f} "
                      f"({sw_tok[si]} tokens)")
        results_dict["sliding_window"] = (sw_loss, sw_tok)

    # ── Comparison table ──
    max_step = max(
        max(s_loss.keys()) for s_loss, _ in results_dict.values() if s_loss
    )
    table = format_retrieval_table(results_dict, max_step)
    print(table)

    # ── Save JSON ──
    json_results = {
        "config": {
            "seq_len": seq_len,
            "step_length": step_length,
            "n_summary_tokens": n_summary_tokens,
            "n_pseudo_tokens": config.get("n_pseudo_tokens", 8),
            "eval_batches": args.eval_batches,
            "retrieval_k": args.retrieval_k,
            "retrieval_strategy": args.retrieval_strategy,
            "max_bank_size": args.max_bank_size,
            "window_tokens": args.window_tokens,
        },
        "memory": memory_stats,
        "per_step": {},
    }
    for si in range(max_step + 1):
        entry = {}
        for cond_name, (s_loss, s_tok) in results_dict.items():
            l = s_loss.get(si, 0)
            t = s_tok.get(si, 0)
            entry[f"{cond_name}_ppl"] = math.exp(l / t) if t > 0 else None
            entry[f"{cond_name}_tokens"] = t
        json_results["per_step"][str(si)] = entry

    # Totals
    json_results["totals"] = {}
    for cond_name, (s_loss, s_tok) in results_dict.items():
        tl = sum(s_loss.values())
        tt = sum(s_tok.values())
        json_results["totals"][f"{cond_name}_ppl"] = (
            math.exp(tl / tt) if tt > 0 else None
        )

    results_dir = Path(config.get("results_dir", "./results-retrieval"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / (
        f"retrieval_{args.retrieval_strategy}_k{args.retrieval_k}"
        f"_seqlen{seq_len}.json"
    )
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
