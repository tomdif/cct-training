"""
Per-step perplexity diagnostic: separates "summaries don't help" from
"summaries help but perplexity dilutes the signal."

Runs THREE conditions with per-step PPL breakdown:
  A. Baseline (full attention) — per-step PPL with full prior context
  B. Sequential WITH summaries — CCT deployment with summary injection
  C. Sequential WITHOUT summaries — each step in isolation, no prior context

The key comparison:
  - B < C at step N means summaries help at distance N
  - A < B gap growing with N means summary quality degrades with distance
  - C getting worse with N means the model genuinely needs cross-step info

Usage:
  python scripts/eval_per_step.py \
    --config configs/tier3_410m_seq_s200.yaml \
    --cct-checkpoint ./checkpoints-tier3-seq-s200/cct-final.pt \
    --baseline-dir ./checkpoints-tier3-seq-s200/baseline-final \
    --eval-batches 200 --seq-len 2048
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
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
from src.model.summary_adapter import SummaryAdapter
from src.model.summary_logit_bias import SummaryLogitBias
from src.model.pseudo_token_decoder import PseudoTokenDecoder


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


def compute_per_step_baseline(model, dataloader, step_token_id, device, max_batches):
    """Full-attention baseline with per-step PPL breakdown."""
    model.eval()
    step_loss = defaultdict(float)
    step_tokens = defaultdict(int)
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            seq_len = input_ids.shape[1]

            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(seq_len, boundary_positions)

            outputs = model(input_ids)
            logits = outputs.logits  # (B, S, V)

            for step_idx, (start, end) in enumerate(step_ranges):
                if end - start <= 1:
                    continue
                # Predict tokens[start+1:end] from logits[start:end-1]
                # But for step_idx > 0, we can also predict the first token
                # of this step from the last token of the previous step
                if step_idx == 0:
                    s_logits = logits[:, start:end - 1, :]
                    s_labels = input_ids[:, start + 1:end].clone()
                else:
                    # Include cross-boundary prediction
                    s_logits = logits[:, start - 1:end - 1, :]
                    s_labels = input_ids[:, start:end].clone()

                s_labels[s_labels == step_token_id] = -100

                loss = torch.nn.functional.cross_entropy(
                    s_logits.reshape(-1, s_logits.size(-1)),
                    s_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (s_labels.reshape(-1) != -100).sum().item()
                step_loss[step_idx] += loss.item()
                step_tokens[step_idx] += n_valid

            n_batches += 1
            if n_batches % 50 == 0:
                total_l = sum(step_loss.values())
                total_t = sum(step_tokens.values())
                avg = total_l / total_t if total_t > 0 else 0
                print(f"  batch {n_batches}/{max_batches} | avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    return step_loss, step_tokens, n_batches


def compute_per_step_sequential(
    model, commitment_head, up_project, dataloader,
    step_token_id, device, max_batches,
    n_summary_tokens=1, use_summaries=True,
    summary_adapter=None,
    summary_logit_bias=None,
    prefix_generator=None,
    pseudo_decoder=None,
    config=None,
):
    """Sequential eval with per-step PPL breakdown.

    use_summaries=True: normal sequential with summary injection
    use_summaries=False: each step in isolation (no prior context)
    """
    model.eval()
    commitment_head.eval()
    step_loss = defaultdict(float)
    step_tokens = defaultdict(int)
    n_batches = 0
    K = n_summary_tokens

    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.gpt_neox.embed_in

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
            committed_summaries = []

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
                step_past_kv = None  # set by KV prefix path if active

                if summary_adapter is not None and use_summaries:
                    # Adapter path: no prepended tokens, set summary bank
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    if n_prior_steps > 0:
                        bank = torch.stack(committed_summaries, dim=0).transpose(0, 1)
                        if bank.dim() > 3:
                            B_s, N_s, K_s, D_s = bank.shape
                            bank = bank.view(B_s, N_s * K_s, D_s)
                        summary_adapter.set_summary_bank(bank)
                elif summary_logit_bias is not None and use_summaries:
                    # Logit bias path: no prepended tokens, bias applied post-forward
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                elif prefix_generator is not None and use_summaries:
                    # KV prefix path: generate past_key_values, no prepended tokens
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    if n_prior_steps > 0:
                        step_past_kv = prefix_generator(
                            committed_summaries[:n_prior_steps],
                            boundary_positions[:n_prior_steps],
                            batch_size,
                        )
                        # Cast KV cache to model dtype (prefix gen runs in float32)
                        for layer in step_past_kv.layers:
                            layer.keys = layer.keys.to(step_embeds.dtype)
                            layer.values = layer.values.to(step_embeds.dtype)
                elif pseudo_decoder is not None and use_summaries and n_prior_steps > 0:
                    # Pseudo-token path: decode each summary into multiple tokens
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
                        bp = boundary_positions[s_idx]
                        for p in range(P):
                            summary_pos_list.append(bp - P + 1 + p)
                elif use_summaries and n_prior_steps > 0:
                    # Attention path: prepend summary tokens
                    n_prepend = n_prior_steps * K
                    summary_embeds_list = []
                    summary_pos_list = []
                    for s_idx in range(n_prior_steps):
                        s = committed_summaries[s_idx]
                        if K > 1:
                            up_proj = up_project(s).to(step_embeds.dtype)
                        else:
                            up_proj = up_project(s).unsqueeze(1).to(step_embeds.dtype)
                        summary_embeds_list.append(up_proj)
                        bp = boundary_positions[s_idx]
                        for k in range(K):
                            summary_pos_list.append(bp - K + 1 + k)

                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)
                    summary_positions = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([summary_positions, step_positions], dim=1)
                else:
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    past_key_values=step_past_kv,
                )
                if summary_adapter is not None:
                    summary_adapter.clear()

                # Apply logit bias from summary bank
                if summary_logit_bias is not None and use_summaries and len(committed_summaries) > 0:
                    if config is not None and config.get("last_summary_only", False):
                        lb_bank = committed_summaries[-1].unsqueeze(0).transpose(0, 1)
                    else:
                        lb_bank = torch.stack(committed_summaries, dim=0).transpose(0, 1)
                    if lb_bank.dim() > 3:
                        B_s, N_s, K_s, D_s = lb_bank.shape
                        lb_bank = lb_bank.view(B_s, N_s * K_s, D_s)
                    bias = summary_logit_bias(lb_bank)  # (B, V)
                    outputs.logits = outputs.logits + bias.unsqueeze(1)

                # Commitment summary for non-last steps (always, for both conditions)
                is_last_step = (step_idx == len(step_ranges) - 1)
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][:, n_prepend:, :]
                    prev_sum = committed_summaries[-1] if committed_summaries else None
                    summary = commitment_head(step_hidden.float(), prev_summary=prev_sum)
                    committed_summaries.append(summary)

                # Loss on step tokens only
                step_logits = outputs.logits[:, n_prepend:n_prepend + step_len - 1, :]
                step_labels = step_token_ids[:, 1:].clone()
                step_labels[step_labels == step_token_id] = -100

                # Cross-boundary prediction (first token of step from last summary)
                if n_prepend > 0:
                    cross_logit = outputs.logits[:, n_prepend - 1:n_prepend, :]
                    first_label = step_token_ids[:, 0:1].clone()
                    first_label[first_label == step_token_id] = -100
                    step_logits = torch.cat([cross_logit, step_logits], dim=1)
                    step_labels = torch.cat([first_label, step_labels], dim=1)

                loss = torch.nn.functional.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    step_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (step_labels.reshape(-1) != -100).sum().item()
                step_loss[step_idx] += loss.item()
                step_tokens[step_idx] += n_valid

            n_batches += 1
            if n_batches % 50 == 0:
                total_l = sum(step_loss.values())
                total_t = sum(step_tokens.values())
                avg = total_l / total_t if total_t > 0 else 0
                print(f"  batch {n_batches}/{max_batches} | avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    return step_loss, step_tokens, n_batches


def format_ppl_table(baseline, with_summary, no_summary, max_step):
    """Format per-step PPL comparison table."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("PER-STEP PERPLEXITY BREAKDOWN")
    lines.append("=" * 80)
    lines.append(f"  {'Step':>4}  {'Baseline':>10}  {'W/ Summary':>10}  {'No Summary':>10}  {'Sum Benefit':>11}  {'vs Base':>10}")
    lines.append(f"  {'':>4}  {'(full ctx)':>10}  {'(CCT)':>10}  {'(isolated)':>10}  {'(C-B)':>11}  {'(B-A)':>10}")
    lines.append("-" * 80)

    for step_idx in range(max_step + 1):
        bl_loss, bl_tok = baseline
        ws_loss, ws_tok = with_summary
        ns_loss, ns_tok = no_summary

        bl_l = bl_loss.get(step_idx, 0)
        bl_t = bl_tok.get(step_idx, 0)
        ws_l = ws_loss.get(step_idx, 0)
        ws_t = ws_tok.get(step_idx, 0)
        ns_l = ns_loss.get(step_idx, 0)
        ns_t = ns_tok.get(step_idx, 0)

        bl_ppl = math.exp(bl_l / bl_t) if bl_t > 0 else float('nan')
        ws_ppl = math.exp(ws_l / ws_t) if ws_t > 0 else float('nan')
        ns_ppl = math.exp(ns_l / ns_t) if ns_t > 0 else float('nan')

        # Summary benefit = no_summary_ppl - with_summary_ppl (positive = summaries help)
        benefit = ns_ppl - ws_ppl if (ws_t > 0 and ns_t > 0) else float('nan')
        # vs baseline = with_summary_ppl - baseline_ppl (positive = CCT is worse)
        vs_base = ws_ppl - bl_ppl if (ws_t > 0 and bl_t > 0) else float('nan')

        tok_str = f"({bl_t}t)" if bl_t > 0 else ""
        lines.append(
            f"  {step_idx:>4}  {bl_ppl:>10.2f}  {ws_ppl:>10.2f}  {ns_ppl:>10.2f}  {benefit:>+11.2f}  {vs_base:>+10.2f}  {tok_str}"
        )

    # Totals
    bl_total_l = sum(baseline[0].values())
    bl_total_t = sum(baseline[1].values())
    ws_total_l = sum(with_summary[0].values())
    ws_total_t = sum(with_summary[1].values())
    ns_total_l = sum(no_summary[0].values())
    ns_total_t = sum(no_summary[1].values())

    bl_total_ppl = math.exp(bl_total_l / bl_total_t) if bl_total_t > 0 else float('nan')
    ws_total_ppl = math.exp(ws_total_l / ws_total_t) if ws_total_t > 0 else float('nan')
    ns_total_ppl = math.exp(ns_total_l / ns_total_t) if ns_total_t > 0 else float('nan')

    lines.append("-" * 80)
    lines.append(
        f"  {'ALL':>4}  {bl_total_ppl:>10.2f}  {ws_total_ppl:>10.2f}  {ns_total_ppl:>10.2f}  "
        f"{ns_total_ppl - ws_total_ppl:>+11.2f}  {ws_total_ppl - bl_total_ppl:>+10.2f}"
    )
    lines.append("")

    # Interpretation
    lines.append("INTERPRETATION:")
    lines.append("  Sum Benefit (C-B): positive = summaries improve over no-context")
    lines.append("  vs Base (B-A):     positive = CCT is worse than full attention")
    lines.append("  If Sum Benefit grows with step => summaries help more at distance")
    lines.append("  If Sum Benefit ~ 0 everywhere => summaries provide no value")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Per-step PPL diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: from config)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seq_len = args.seq_len or config["seq_len"]
    step_length = config["step_length"]
    n_summary_tokens = config.get("n_summary_tokens", 1)

    print(f"Config: seq_len={seq_len}, step_length={step_length}, "
          f"n_summary_tokens={n_summary_tokens}")
    print(f"Expected steps per sequence: ~{seq_len // (step_length + 1)}")

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

    # ══════════════════════════════════════════════════════════
    # CONDITION A: BASELINE (full attention, per-step breakdown)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION A: BASELINE (full attention)")
    print("=" * 60)
    baseline_oom = False
    try:
        baseline_model = AutoModelForCausalLM.from_pretrained(
            args.baseline_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).to(device)
        baseline_model.resize_token_embeddings(len(tokenizer))

        eval_dataset_a = StreamingEvalDataset(
            dataset_name=config.get("validation_dataset", config["dataset"]),
            split=config.get("validation_split", "train"),
            tokenizer=tokenizer, seq_len=seq_len,
            step_token_id=step_token_id, step_length=step_length,
            skip_examples=50000,
        )
        loader_a = DataLoader(eval_dataset_a, batch_size=batch_size)
        bl_loss, bl_tok, bl_n = compute_per_step_baseline(
            baseline_model, loader_a, step_token_id, device, args.eval_batches
        )
        total_bl = sum(bl_loss.values()) / sum(bl_tok.values())
        print(f"\n  Baseline total: loss={total_bl:.4f} ppl={math.exp(total_bl):.2f}")
        for si in sorted(bl_loss.keys()):
            if bl_tok[si] > 0:
                ppl = math.exp(bl_loss[si] / bl_tok[si])
                print(f"    step {si}: ppl={ppl:.2f} ({bl_tok[si]} tokens)")

        del baseline_model
    except torch.cuda.OutOfMemoryError:
        print(f"\n  *** BASELINE OOM: Cannot process seq_len={seq_len} with full attention ***")
        print(f"  *** Attention matrix alone requires ~{seq_len**2 * 16 * 2 / 1e9:.1f} GB per layer ***")
        baseline_oom = True
        bl_loss, bl_tok = {}, {}
        # Try to clean up partial allocation
        try:
            del baseline_model
        except NameError:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Load CCT model + components
    print("\nLoading CCT model...")
    cct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    cct_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)

    # Apply LoRA wrapping if checkpoint was trained with LoRA
    if config.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=0.0,
            target_modules=config.get("lora_target_modules", ["query_key_value"]),
            layers_to_transform=list(range(
                config.get("lora_layers_min", 12),
                config.get("lora_layers_max", 23) + 1,
            )),
            bias="none",
        )
        cct_model = get_peft_model(cct_model, lora_config)

    # Cast checkpoint weights to model dtype (LoRA weights may be float32 from AMP training)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cct_state = {k: v.to(model_dtype) if v.is_floating_point() else v for k, v in ckpt["model_state_dict"].items()}
    cct_model.load_state_dict(cct_state)

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

    summary_buffer = SummaryBuffer(
        d_summary=config["d_summary"],
        d_model=config["d_model"],
        device=device,
        decoder_type=config.get("decoder_type", "linear"),
        decoder_bottleneck=config.get("decoder_bottleneck", None),
    )
    summary_buffer.up_project.load_state_dict(ckpt["summary_up_project_state_dict"])
    up_project = summary_buffer.up_project

    # Load summary adapter if present
    summary_adapter = None
    if ckpt.get("summary_adapter_state_dict") is not None:
        n_layers = cct_model.config.num_hidden_layers
        adapter_d = config.get("adapter_d", 64)
        summary_adapter = SummaryAdapter(
            d_model=config["d_model"],
            d_summary=config["d_summary"],
            n_layers=n_layers,
            d_adapter=adapter_d,
            device=device,
        )
        summary_adapter.load_state_dict(ckpt["summary_adapter_state_dict"])
        summary_adapter.register_hooks(cct_model)
        print(f"  Loaded summary adapter: {summary_adapter.param_count():,} params")

    # Load summary logit bias if present
    summary_logit_bias = None
    if ckpt.get("summary_logit_bias_state_dict") is not None:
        summary_logit_bias = SummaryLogitBias(
            d_summary=config["d_summary"],
            vocab_size=cct_model.config.vocab_size,
            hidden_dim=config.get("logit_bias_hidden_dim", 256),
            device=device,
        )
        summary_logit_bias.load_state_dict(ckpt["summary_logit_bias_state_dict"])
        print(f"  Loaded summary logit bias: {summary_logit_bias.param_count():,} params")

    # Load pseudo-token decoder if present
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

    # Load prefix generator if present
    prefix_generator = None
    if ckpt.get("prefix_generator_state_dict") is not None:
        from src.model.prefix_generator import PrefixKVGenerator
        prefix_generator = PrefixKVGenerator(
            d_summary=config["d_summary"],
            n_layers=cct_model.config.num_hidden_layers,
            n_heads=cct_model.config.num_attention_heads,
            d_head=config["d_model"] // cct_model.config.num_attention_heads,
            hidden_dim=config.get("kv_prefix_hidden", 512),
            active_layers_min=config.get("lora_layers_min", 12),
        ).to(device)
        prefix_generator.load_state_dict(ckpt["prefix_generator_state_dict"])
        prefix_generator.eval()
        print(f"  Loaded prefix generator: {prefix_generator.param_count():,} params")

    # ══════════════════════════════════════════════════════════
    # CONDITION B: SEQUENTIAL WITH SUMMARIES
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION B: SEQUENTIAL WITH SUMMARIES")
    print("=" * 60)
    eval_dataset_b = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer, seq_len=seq_len,
        step_token_id=step_token_id, step_length=step_length,
        skip_examples=50000,
    )
    loader_b = DataLoader(eval_dataset_b, batch_size=batch_size)
    ws_loss, ws_tok, ws_n = compute_per_step_sequential(
        cct_model, commitment_head, up_project, loader_b,
        step_token_id, device, args.eval_batches,
        n_summary_tokens=n_summary_tokens, use_summaries=True,
        summary_adapter=summary_adapter,
        summary_logit_bias=summary_logit_bias,
        prefix_generator=prefix_generator,
        pseudo_decoder=pseudo_decoder,
        config=config,
    )
    total_ws = sum(ws_loss.values()) / sum(ws_tok.values())
    print(f"\n  Sequential (w/ summaries) total: loss={total_ws:.4f} ppl={math.exp(total_ws):.2f}")
    for si in sorted(ws_loss.keys()):
        if ws_tok[si] > 0:
            ppl = math.exp(ws_loss[si] / ws_tok[si])
            print(f"    step {si}: ppl={ppl:.2f} ({ws_tok[si]} tokens)")

    # ══════════════════════════════════════════════════════════
    # CONDITION C: SEQUENTIAL WITHOUT SUMMARIES
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION C: SEQUENTIAL WITHOUT SUMMARIES (isolated steps)")
    print("=" * 60)
    eval_dataset_c = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer, seq_len=seq_len,
        step_token_id=step_token_id, step_length=step_length,
        skip_examples=50000,
    )
    loader_c = DataLoader(eval_dataset_c, batch_size=batch_size)
    ns_loss, ns_tok, ns_n = compute_per_step_sequential(
        cct_model, commitment_head, up_project, loader_c,
        step_token_id, device, args.eval_batches,
        n_summary_tokens=n_summary_tokens, use_summaries=False,
        summary_adapter=summary_adapter,
        summary_logit_bias=summary_logit_bias,
        prefix_generator=prefix_generator,
        pseudo_decoder=pseudo_decoder,
        config=config,
    )
    total_ns = sum(ns_loss.values()) / sum(ns_tok.values())
    print(f"\n  Sequential (no summaries) total: loss={total_ns:.4f} ppl={math.exp(total_ns):.2f}")
    for si in sorted(ns_loss.keys()):
        if ns_tok[si] > 0:
            ppl = math.exp(ns_loss[si] / ns_tok[si])
            print(f"    step {si}: ppl={ppl:.2f} ({ns_tok[si]} tokens)")

    if summary_adapter is not None:
        summary_adapter.remove_hooks()
    del cct_model, commitment_head, up_project, summary_buffer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════
    max_step = max(
        max(bl_loss.keys()) if bl_loss else 0,
        max(ws_loss.keys()) if ws_loss else 0,
        max(ns_loss.keys()) if ns_loss else 0,
    )
    table = format_ppl_table(
        (bl_loss, bl_tok), (ws_loss, ws_tok), (ns_loss, ns_tok), max_step
    )
    print(table)

    # Save results
    results = {
        "config": {
            "seq_len": seq_len,
            "step_length": step_length,
            "n_summary_tokens": n_summary_tokens,
            "eval_batches": args.eval_batches,
        },
        "per_step": {},
    }
    for si in range(max_step + 1):
        results["per_step"][str(si)] = {
            "baseline_ppl": math.exp(bl_loss[si] / bl_tok[si]) if bl_tok.get(si, 0) > 0 else None,
            "with_summary_ppl": math.exp(ws_loss[si] / ws_tok[si]) if ws_tok.get(si, 0) > 0 else None,
            "no_summary_ppl": math.exp(ns_loss[si] / ns_tok[si]) if ns_tok.get(si, 0) > 0 else None,
            "baseline_tokens": bl_tok.get(si, 0),
            "with_summary_tokens": ws_tok.get(si, 0),
            "no_summary_tokens": ns_tok.get(si, 0),
        }

    results_dir = Path(config.get("results_dir", "./results-per-step"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"per_step_ppl_seqlen{seq_len}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
