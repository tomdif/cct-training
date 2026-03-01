"""
Experiment 1: Session Resumption

Measures whether the CCT summary bank preserves useful context across a
"session boundary" where the KV cache is discarded.

Process a long document (N steps). Then evaluate PPL on the LAST step
under three conditions:

  1. Cold start:      Last step only, no prior context. Amnesia baseline.
  2. Summary resume:  Last step + pseudo-tokens from saved summary bank.
                      This is CCT persistent memory.
  3. Full attention:  All tokens with full causal attention. Oracle.

The gap between cold start and summary resume = value of persistent memory.
The gap between summary resume and full attention = information lost.

Sliding window gets exactly the cold-start condition at session boundaries —
it has no mechanism to remember earlier steps.

Usage:
  python scripts/eval_session_resumption.py \
    --config configs/tier3_410m_retrieval_P.yaml \
    --cct-checkpoint checkpoints-run-P-retrieval/cct-final.pt \
    --seq-len 8192 --eval-batches 50
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_pipeline import (
    extend_tokenizer,
    annotate_step_boundaries,
    get_step_boundary_positions,
    get_step_ranges,
)
from src.model.commitment_head import CommitmentHead
from src.model.pseudo_token_decoder import PseudoTokenDecoder
from src.model.kv_cache_utils import (
    _cache_to_tuples, _tuples_to_cache, extract_step_kv, merge_kv_caches,
)


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


def load_cct_components(config, checkpoint_path, device):
    """Load model and CCT components from checkpoint."""
    print(f"Loading model: {config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "eager"),
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    step_token_id = extend_tokenizer(tokenizer)

    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.gpt_neox.embed_in

    commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config.get("d_summary", 16),
        d_bottleneck=config.get("d_bottleneck", 64),
        n_summary_tokens=config.get("n_summary_tokens", 1),
        use_tanh=config.get("use_tanh", False),
        use_l2_norm=config.get("use_l2_norm", False),
        recurrent=config.get("recurrent_commitment", False),
    ).to(device)

    pseudo_decoder = None
    if config.get("use_pseudo_tokens", False):
        pseudo_decoder = PseudoTokenDecoder(
            d_summary=config.get("d_summary", 16),
            d_model=config["d_model"],
            n_pseudo_tokens=config.get("n_pseudo_tokens", 8),
            hidden_dim=config.get("pseudo_decoder_hidden", 512),
        ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    commitment_head.load_state_dict(ckpt["commitment_head_state_dict"])
    if pseudo_decoder is not None and "pseudo_decoder_state_dict" in ckpt:
        pseudo_decoder.load_state_dict(ckpt["pseudo_decoder_state_dict"])
    print(f"  Loaded from step {ckpt.get('global_step', '?')}")

    n_layers = model.config.num_hidden_layers
    return model, tokenizer, embed_layer, commitment_head, pseudo_decoder, step_token_id, n_layers


def process_context_steps(
    model, commitment_head, pseudo_decoder, embed_layer,
    input_ids, step_ranges, step_token_id, device, n_layers,
    max_bank_size=8, retrieval_k=2,
):
    """Process all steps EXCEPT the last one. Build up summary bank and KV bank.

    Returns:
        committed_summaries: list of GRU-accumulated summaries
        kv_bank: list of per-step KV caches (most recent max_bank_size)
    """
    batch_size = input_ids.shape[0]
    committed_summaries = []
    kv_bank = []
    K = 1  # n_summary_tokens

    for step_idx, (start, end) in enumerate(step_ranges[:-1]):  # all but last
        step_len = end - start
        if step_len <= 0:
            continue

        step_token_ids = input_ids[:, start:end]
        step_embeds = embed_layer(step_token_ids)
        step_positions = torch.arange(start, end, device=device).unsqueeze(0).expand(batch_size, -1)

        # Pseudo-token prefix from accumulated summaries
        n_prepend = 0
        if pseudo_decoder is not None and len(committed_summaries) > 0:
            P = pseudo_decoder.n_pseudo_tokens
            n_prepend = len(committed_summaries) * P
            summary_embeds_list = []
            for s in committed_summaries:
                if K > 1:
                    s = s[:, 0, :]
                decoded = pseudo_decoder(s).to(step_embeds.dtype)
                summary_embeds_list.append(decoded)
            summary_embeds = torch.cat(summary_embeds_list, dim=1)
            inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)

            summary_pos = []
            for i in range(n_prepend):
                summary_pos.append(max(0, start - n_prepend + i))
            summary_positions = torch.tensor(summary_pos, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            position_ids = torch.cat([summary_positions, step_positions], dim=1)
        else:
            inputs_embeds = step_embeds
            position_ids = step_positions

        # Retrieved KV from recent steps
        retrieved_kv = None
        if len(kv_bank) > 0:
            k = min(retrieval_k, len(kv_bank))
            selected = kv_bank[-k:]  # most recent
            retrieved_kv = merge_kv_caches(selected, n_layers)

        past_kv_arg = _tuples_to_cache(retrieved_kv) if retrieved_kv is not None else None
        outputs = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=True,
            past_key_values=past_kv_arg,
            use_cache=True,
        )

        # Store KV
        if outputs.past_key_values is not None:
            step_kv = extract_step_kv(outputs.past_key_values, step_len, n_layers)
            kv_bank.append(step_kv)
            while len(kv_bank) > max_bank_size:
                evicted = kv_bank.pop(0)
                del evicted

        # Commitment summary (GRU)
        step_hidden = outputs.hidden_states[-1][:, n_prepend:, :]
        prev_sum = committed_summaries[-1] if committed_summaries else None
        summary = commitment_head(step_hidden.float(), prev_summary=prev_sum)
        committed_summaries.append(summary)

    return committed_summaries, kv_bank


def eval_last_step_cold(model, embed_layer, input_ids, step_ranges, step_token_id, device):
    """Condition 1: Cold start — last step only, no prior context."""
    start, end = step_ranges[-1]
    step_len = end - start
    batch_size = input_ids.shape[0]

    step_token_ids = input_ids[:, start:end]
    step_embeds = embed_layer(step_token_ids)
    # Positions start from 0 — cold start has no context
    position_ids = torch.arange(step_len, device=device).unsqueeze(0).expand(batch_size, -1)

    outputs = model(inputs_embeds=step_embeds, position_ids=position_ids)

    logits = outputs.logits[:, :-1, :]
    labels = step_token_ids[:, 1:].clone()
    labels[labels == step_token_id] = -100

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_valid = (labels.reshape(-1) != -100).sum().item()
    return loss.item(), n_valid


def eval_last_step_summary_resume(
    model, embed_layer, pseudo_decoder, commitment_head,
    input_ids, step_ranges, step_token_id, device,
    committed_summaries, kv_bank, n_layers, retrieval_k=2,
):
    """Condition 2: Summary resume — last step + pseudo-tokens from summary bank.
    Also uses retrieved KV if available (full hybrid)."""
    start, end = step_ranges[-1]
    step_len = end - start
    batch_size = input_ids.shape[0]
    K = 1

    step_token_ids = input_ids[:, start:end]
    step_embeds = embed_layer(step_token_ids)
    step_positions = torch.arange(start, end, device=device).unsqueeze(0).expand(batch_size, -1)

    n_prepend = 0
    if pseudo_decoder is not None and len(committed_summaries) > 0:
        P = pseudo_decoder.n_pseudo_tokens
        n_prepend = len(committed_summaries) * P
        summary_embeds_list = []
        for s in committed_summaries:
            if K > 1:
                s = s[:, 0, :]
            decoded = pseudo_decoder(s).to(step_embeds.dtype)
            summary_embeds_list.append(decoded)
        summary_embeds = torch.cat(summary_embeds_list, dim=1)
        inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)

        summary_pos = []
        for i in range(n_prepend):
            summary_pos.append(max(0, start - n_prepend + i))
        summary_positions = torch.tensor(summary_pos, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.cat([summary_positions, step_positions], dim=1)
    else:
        inputs_embeds = step_embeds
        position_ids = step_positions

    # Retrieved KV
    retrieved_kv = None
    if len(kv_bank) > 0:
        k = min(retrieval_k, len(kv_bank))
        selected = kv_bank[-k:]
        retrieved_kv = merge_kv_caches(selected, n_layers)

    past_kv_arg = _tuples_to_cache(retrieved_kv) if retrieved_kv is not None else None
    outputs = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        past_key_values=past_kv_arg,
    )

    # Loss on real tokens only (skip pseudo prefix)
    logits = outputs.logits[:, n_prepend:n_prepend + step_len - 1, :]
    labels = step_token_ids[:, 1:].clone()
    labels[labels == step_token_id] = -100

    # Cross-boundary from last pseudo-token
    if n_prepend > 0:
        cross_logit = outputs.logits[:, n_prepend - 1:n_prepend, :]
        first_label = step_token_ids[:, 0:1].clone()
        first_label[first_label == step_token_id] = -100
        logits = torch.cat([cross_logit, logits], dim=1)
        labels = torch.cat([first_label, labels], dim=1)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_valid = (labels.reshape(-1) != -100).sum().item()
    return loss.item(), n_valid


def eval_last_step_summary_only(
    model, embed_layer, pseudo_decoder,
    input_ids, step_ranges, step_token_id, device,
    committed_summaries,
):
    """Condition 2b: Summary only — pseudo-tokens but NO retrieved KV.
    Pure persistent memory without any KV cache."""
    start, end = step_ranges[-1]
    step_len = end - start
    batch_size = input_ids.shape[0]
    K = 1

    step_token_ids = input_ids[:, start:end]
    step_embeds = embed_layer(step_token_ids)
    step_positions = torch.arange(start, end, device=device).unsqueeze(0).expand(batch_size, -1)

    n_prepend = 0
    if pseudo_decoder is not None and len(committed_summaries) > 0:
        P = pseudo_decoder.n_pseudo_tokens
        n_prepend = len(committed_summaries) * P
        summary_embeds_list = []
        for s in committed_summaries:
            if K > 1:
                s = s[:, 0, :]
            decoded = pseudo_decoder(s).to(step_embeds.dtype)
            summary_embeds_list.append(decoded)
        summary_embeds = torch.cat(summary_embeds_list, dim=1)
        inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)

        summary_pos = []
        for i in range(n_prepend):
            summary_pos.append(max(0, start - n_prepend + i))
        summary_positions = torch.tensor(summary_pos, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.cat([summary_positions, step_positions], dim=1)
    else:
        inputs_embeds = step_embeds
        position_ids = step_positions

    outputs = model(inputs_embeds=inputs_embeds, position_ids=position_ids)

    logits = outputs.logits[:, n_prepend:n_prepend + step_len - 1, :]
    labels = step_token_ids[:, 1:].clone()
    labels[labels == step_token_id] = -100

    if n_prepend > 0:
        cross_logit = outputs.logits[:, n_prepend - 1:n_prepend, :]
        first_label = step_token_ids[:, 0:1].clone()
        first_label[first_label == step_token_id] = -100
        logits = torch.cat([cross_logit, logits], dim=1)
        labels = torch.cat([first_label, labels], dim=1)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_valid = (labels.reshape(-1) != -100).sum().item()
    return loss.item(), n_valid


def eval_last_step_full_attention(model, embed_layer, input_ids, step_ranges, step_token_id, device):
    """Condition 3: Full attention — all tokens, standard causal mask. Oracle."""
    start, end = step_ranges[-1]
    batch_size = input_ids.shape[0]

    # Feed ALL tokens up to the end of the last step
    all_token_ids = input_ids[:, :end]

    # Strip step boundary tokens for clean full-attention baseline
    # Replace step tokens with a common token (pad or repeat prior)
    clean_ids = all_token_ids.clone()
    clean_ids[clean_ids == step_token_id] = 1  # replace with token 1 (arbitrary common token)

    # Full-sequence forward pass
    seq_len_full = clean_ids.shape[1]
    position_ids = torch.arange(seq_len_full, device=device).unsqueeze(0).expand(batch_size, -1)

    outputs = model(input_ids=clean_ids, position_ids=position_ids)

    # Only measure loss on the LAST STEP tokens (same tokens as other conditions)
    logits = outputs.logits[:, start:end - 1, :]
    labels = input_ids[:, start + 1:end].clone()
    labels[labels == step_token_id] = -100

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_valid = (labels.reshape(-1) != -100).sum().item()
    return loss.item(), n_valid


def main():
    parser = argparse.ArgumentParser(description="Session Resumption Experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--max-bank-size", type=int, default=8)
    parser.add_argument("--retrieval-k", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer, embed_layer, commitment_head, pseudo_decoder, step_token_id, n_layers = \
        load_cct_components(config, args.cct_checkpoint, device)

    step_length = config.get("step_length", 400)

    # Dataset
    eval_ds = StreamingEvalDataset(
        config["dataset"], config.get("validation_split", "train"),
        tokenizer, args.seq_len, step_token_id, step_length,
    )
    dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=1)

    # Accumulators for each condition
    conditions = {
        "cold_start": {"loss": 0.0, "tokens": 0},
        "summary_only": {"loss": 0.0, "tokens": 0},
        "summary_resume": {"loss": 0.0, "tokens": 0},
        "full_attention": {"loss": 0.0, "tokens": 0},
    }

    n_batches = 0
    n_steps_total = 0

    print(f"\n{'='*70}")
    print(f"  SESSION RESUMPTION EXPERIMENT")
    print(f"  seq_len={args.seq_len}, bank_size={args.max_bank_size}, k={args.retrieval_k}")
    print(f"  Process N-1 steps (build context), eval PPL on last step")
    print(f"{'='*70}\n")

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= args.eval_batches:
                break

            input_ids = batch.to(device)
            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(input_ids.shape[1], boundary_positions)

            if len(step_ranges) < 3:
                continue  # need at least 3 steps for meaningful test

            n_context_steps = len(step_ranges) - 1
            n_steps_total += n_context_steps

            # ---- Phase 1: Process context steps (build summary bank) ----
            committed_summaries, kv_bank = process_context_steps(
                model, commitment_head, pseudo_decoder, embed_layer,
                input_ids, step_ranges, step_token_id, device, n_layers,
                max_bank_size=args.max_bank_size,
                retrieval_k=args.retrieval_k,
            )

            # ---- Phase 2: Eval last step under each condition ----

            # Condition 1: Cold start (amnesia)
            loss, tokens = eval_last_step_cold(
                model, embed_layer, input_ids, step_ranges, step_token_id, device,
            )
            conditions["cold_start"]["loss"] += loss
            conditions["cold_start"]["tokens"] += tokens

            # Condition 2b: Summary only (pseudo-tokens, no KV retrieval)
            loss, tokens = eval_last_step_summary_only(
                model, embed_layer, pseudo_decoder,
                input_ids, step_ranges, step_token_id, device,
                committed_summaries,
            )
            conditions["summary_only"]["loss"] += loss
            conditions["summary_only"]["tokens"] += tokens

            # Condition 2: Summary resume (pseudo-tokens + retrieved KV)
            loss, tokens = eval_last_step_summary_resume(
                model, embed_layer, pseudo_decoder, commitment_head,
                input_ids, step_ranges, step_token_id, device,
                committed_summaries, kv_bank, n_layers,
                retrieval_k=args.retrieval_k,
            )
            conditions["summary_resume"]["loss"] += loss
            conditions["summary_resume"]["tokens"] += tokens

            # Condition 3: Full attention (oracle)
            loss, tokens = eval_last_step_full_attention(
                model, embed_layer, input_ids, step_ranges, step_token_id, device,
            )
            conditions["full_attention"]["loss"] += loss
            conditions["full_attention"]["tokens"] += tokens

            # Clean up
            del committed_summaries, kv_bank
            torch.cuda.empty_cache()

            n_batches += 1
            if n_batches % 10 == 0:
                print(f"  batch {n_batches}/{args.eval_batches}")

    # ---- Results ----
    print(f"\n{'='*70}")
    print(f"  SESSION RESUMPTION RESULTS")
    print(f"  {n_batches} sequences, {args.seq_len} tokens each")
    print(f"  Context: {n_steps_total/n_batches:.0f} steps avg, then session boundary")
    print(f"  Metric: PPL on last step (after KV cache discarded)")
    print(f"{'='*70}\n")

    results = {}
    for name, data in conditions.items():
        if data["tokens"] > 0:
            avg_loss = data["loss"] / data["tokens"]
            ppl = math.exp(avg_loss)
        else:
            ppl = float("nan")
        results[name] = ppl

    cold = results["cold_start"]
    summary_only = results["summary_only"]
    hybrid = results["summary_resume"]
    oracle = results["full_attention"]

    # Memory preservation ratio: how much of the oracle's advantage does CCT preserve?
    oracle_gap = cold - oracle  # total possible improvement
    cct_gap = cold - hybrid     # actual improvement from CCT
    preservation = (cct_gap / oracle_gap * 100) if oracle_gap > 0 else float("nan")

    summary_gap = cold - summary_only
    summary_preservation = (summary_gap / oracle_gap * 100) if oracle_gap > 0 else float("nan")

    print(f"  {'Condition':<25} {'PPL':>8}  {'vs Cold':>10}  Notes")
    print(f"  {'-'*65}")
    print(f"  {'Cold start (amnesia)':<25} {cold:>8.2f}  {'---':>10}  No prior context")
    print(f"  {'Summary only (pseudo)':<25} {summary_only:>8.2f}  {cold - summary_only:>+10.2f}  Pseudo-tokens, no KV")
    print(f"  {'Summary resume (hybrid)':<25} {hybrid:>8.2f}  {cold - hybrid:>+10.2f}  Pseudo-tokens + KV")
    print(f"  {'Full attention (oracle)':<25} {oracle:>8.2f}  {cold - oracle:>+10.2f}  All tokens, standard attn")
    print()
    print(f"  Oracle advantage over cold start:  {oracle_gap:+.2f} PPL")
    print(f"  Summary-only preserves:            {summary_preservation:.1f}% of oracle advantage")
    print(f"  Hybrid (summary+KV) preserves:     {preservation:.1f}% of oracle advantage")
    print()
    print(f"  KEY METRIC: At session boundaries, CCT preserves {preservation:.0f}% of what")
    print(f"  full attention provides. Sliding window preserves 0%.")

    # Save results
    results_dir = Path(config.get("results_dir", "./results-session-resumption"))
    results_dir.mkdir(parents=True, exist_ok=True)
    outfile = results_dir / f"session_resumption_{args.seq_len}.json"
    with open(outfile, "w") as f:
        json.dump({
            "seq_len": args.seq_len,
            "n_batches": n_batches,
            "n_context_steps_avg": n_steps_total / n_batches,
            "cold_start_ppl": cold,
            "summary_only_ppl": summary_only,
            "summary_resume_ppl": hybrid,
            "full_attention_ppl": oracle,
            "oracle_advantage": oracle_gap,
            "summary_only_preservation_pct": summary_preservation,
            "hybrid_preservation_pct": preservation,
        }, f, indent=2)
    print(f"\n  Saved: {outfile}")


if __name__ == "__main__":
    main()
