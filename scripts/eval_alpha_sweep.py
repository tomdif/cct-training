"""
Post-hoc logit bias alpha sweep.

Sweeps a scalar multiplier on the logit bias output to find the optimal
trade-off between summary benefit and deployment delta.

For each alpha in [0.0, 0.1, 0.2, ..., 1.0]:
  - Runs per-step sequential eval with bias scaled by alpha
  - Reports deployment PPL, summary benefit per step, and deployment delta

Usage:
  python scripts/eval_alpha_sweep.py \
    --config configs/tier3_410m_logit_bias_F.yaml \
    --cct-checkpoint ./checkpoints-tier3-logit-bias-F/cct-final.pt \
    --baseline-dir ./checkpoints-tier3-seq-s200/baseline-final \
    --eval-batches 50
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_pipeline import (
    extend_tokenizer,
    get_step_boundary_positions,
    get_step_ranges,
)
from src.model.commitment_head import CommitmentHead
from src.model.summary_buffer import SummaryBuffer
from src.model.summary_logit_bias import SummaryLogitBias


class StreamingEvalDataset(torch.utils.data.IterableDataset):
    """Deterministic eval data — skips first `skip_examples` to avoid train overlap."""

    def __init__(self, dataset_name, split, tokenizer, seq_len, step_token_id,
                 step_length, skip_examples=50000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.step_token_id = step_token_id
        self.step_length = step_length
        self.skip_examples = skip_examples
        self.dataset_name = dataset_name
        self.split = split

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        ds = ds.skip(self.skip_examples)
        buffer = []
        for example in ds:
            tokens = self.tokenizer.encode(example["text"], add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                # Insert step tokens at fixed boundaries
                annotated = []
                for i, t in enumerate(chunk):
                    if i > 0 and i % self.step_length == 0:
                        annotated.append(self.step_token_id)
                    annotated.append(t)
                yield torch.tensor(annotated[: self.seq_len], dtype=torch.long)


def compute_per_step_with_alpha(
    model, commitment_head, up_project, dataloader,
    step_token_id, device, max_batches,
    summary_logit_bias, alpha,
    n_summary_tokens=1, use_summaries=True,
):
    """Sequential eval with per-step PPL, applying bias * alpha."""
    model.eval()
    commitment_head.eval()

    step_loss = {}
    step_tokens = {}
    n_batches = 0

    embed_layer = model.gpt_neox.embed_in
    config = model.config
    K = n_summary_tokens

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device) if batch.dim() == 2 else batch.unsqueeze(0).to(device)
            batch_size = input_ids.size(0)

            step_boundaries = get_step_boundary_positions(input_ids[0], step_token_id)
            step_ranges = get_step_ranges(input_ids.size(1), step_boundaries)
            if len(step_ranges) < 2:
                continue

            committed_summaries = []
            boundary_positions = []

            for step_idx, (start, end) in enumerate(step_ranges):
                step_len = end - start
                if step_len <= 1:
                    continue

                step_token_ids = input_ids[:, start:end]

                # Build input: prepend summary tokens if available
                n_prepend = 0
                if use_summaries and committed_summaries and summary_logit_bias is not None:
                    # Logit bias path: no prepend, bias applied to logits
                    step_embeds = embed_layer(step_token_ids)
                    step_positions = torch.arange(
                        start, end, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                elif use_summaries and committed_summaries:
                    # Attention prepend path
                    summary_embeds_list = []
                    summary_pos_list = []
                    for j, s in enumerate(committed_summaries):
                        if K > 1:
                            up = up_project(s)
                        else:
                            up = up_project(s).unsqueeze(1)
                        summary_embeds_list.append(up.to(embed_layer.weight.dtype))
                        bp = boundary_positions[j]
                        for k in range(K):
                            summary_pos_list.append(bp - K + 1 + k)
                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    n_prepend = summary_embeds.size(1)
                    token_embeds = embed_layer(step_token_ids)
                    step_embeds = torch.cat([summary_embeds, token_embeds], dim=1)
                    summary_pos = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    token_pos = torch.arange(
                        start, end, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    step_positions = torch.cat([summary_pos, token_pos], dim=1)
                else:
                    step_embeds = embed_layer(step_token_ids)
                    step_positions = torch.arange(
                        start, end, device=device
                    ).unsqueeze(0).expand(batch_size, -1)

                outputs = model(
                    inputs_embeds=step_embeds,
                    position_ids=step_positions,
                    output_hidden_states=True,
                )

                # Apply scaled logit bias
                if summary_logit_bias is not None and use_summaries and len(committed_summaries) > 0:
                    lb_bank = torch.stack(committed_summaries, dim=0).transpose(0, 1)
                    if lb_bank.dim() > 3:
                        B_s, N_s, K_s, D_s = lb_bank.shape
                        lb_bank = lb_bank.view(B_s, N_s * K_s, D_s)
                    bias = summary_logit_bias(lb_bank)  # (B, V)
                    outputs.logits = outputs.logits + alpha * bias.unsqueeze(1)

                # Commitment summary for non-last steps (always computed)
                is_last_step = (step_idx == len(step_ranges) - 1)
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][:, n_prepend:, :]
                    summary = commitment_head(step_hidden.float())
                    committed_summaries.append(summary)
                    if step_boundaries and step_idx < len(step_boundaries):
                        boundary_positions.append(step_boundaries[step_idx])

                # Loss on step tokens
                step_logits = outputs.logits[:, n_prepend:n_prepend + step_len - 1, :]
                step_labels = step_token_ids[:, 1:].clone()
                step_labels[step_labels == step_token_id] = -100

                # Cross-boundary prediction
                if n_prepend > 0:
                    cross_logit = outputs.logits[:, n_prepend - 1:n_prepend, :]
                    first_label = step_token_ids[:, 0:1].clone()
                    first_label[first_label == step_token_id] = -100
                    step_logits = torch.cat([cross_logit, step_logits], dim=1)
                    step_labels = torch.cat([first_label, step_labels], dim=1)

                loss = nn.functional.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    step_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (step_labels.reshape(-1) != -100).sum().item()

                if n_valid > 0:
                    if step_idx not in step_loss:
                        step_loss[step_idx] = 0.0
                        step_tokens[step_idx] = 0
                    step_loss[step_idx] += loss.item()
                    step_tokens[step_idx] += n_valid

            n_batches += 1

    return step_loss, step_tokens, n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated alpha values to sweep")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_token_id = extend_tokenizer(tokenizer)

    # ═══════════════════════════════════════════════
    # Load baseline model for reference PPL
    # ═══════════════════════════════════════════════
    print("Loading baseline model...")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_dir,
        attn_implementation="eager",
        torch_dtype=model_dtype,
    ).to(device)
    baseline_model.resize_token_embeddings(len(tokenizer))
    baseline_model.eval()

    eval_dataset = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )

    # Baseline per-step
    print("\nComputing baseline per-step PPL...")
    from src.model.commitment_head import CommitmentHead
    bl_commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config["d_summary"],
        d_bottleneck=config["d_bottleneck"],
    ).to(device)
    bl_up = nn.Linear(config["d_summary"], config["d_model"]).to(device)

    bl_loader = DataLoader(eval_dataset, batch_size=config.get("batch_size", 2))
    bl_step_loss, bl_step_tokens, bl_batches = compute_per_step_with_alpha(
        baseline_model, bl_commitment_head, bl_up, bl_loader,
        step_token_id, device, args.eval_batches,
        summary_logit_bias=None, alpha=0.0,
        use_summaries=False,
    )
    baseline_ppl = {}
    for si in sorted(bl_step_loss.keys()):
        if bl_step_tokens[si] > 0:
            baseline_ppl[si] = math.exp(bl_step_loss[si] / bl_step_tokens[si])
    print(f"  Baseline per-step PPL: {', '.join(f'step {k}: {v:.2f}' for k, v in baseline_ppl.items())}")

    # Overall baseline PPL
    total_bl_loss = sum(bl_step_loss.values())
    total_bl_tokens = sum(bl_step_tokens.values())
    bl_ppl_overall = math.exp(total_bl_loss / total_bl_tokens) if total_bl_tokens > 0 else 0.0
    print(f"  Baseline overall PPL: {bl_ppl_overall:.2f}")

    del baseline_model, bl_commitment_head, bl_up
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════
    # Load CCT model
    # ═══════════════════════════════════════════════
    print("\nLoading CCT model...")
    cct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=model_dtype,
    ).to(device)
    cct_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)

    # Cast model state dict to model dtype (handles LoRA float32 weights)
    cct_state = {k: v.to(model_dtype) if v.is_floating_point() else v
                 for k, v in ckpt["model_state_dict"].items()}
    cct_model.load_state_dict(cct_state)
    cct_model.eval()

    commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config["d_summary"],
        d_bottleneck=config["d_bottleneck"],
        use_tanh=config.get("use_tanh", False),
        use_l2_norm=config.get("use_l2_norm", False),
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

    # Load logit bias
    summary_logit_bias = None
    if ckpt.get("summary_logit_bias_state_dict") is not None:
        summary_logit_bias = SummaryLogitBias(
            d_summary=config["d_summary"],
            vocab_size=cct_model.config.vocab_size,
            hidden_dim=config.get("logit_bias_hidden_dim", 256),
            device=device,
        )
        summary_logit_bias.load_state_dict(ckpt["summary_logit_bias_state_dict"])
        summary_logit_bias.eval()
        log_mag = summary_logit_bias.log_magnitude.item()
        print(f"  Loaded summary logit bias: {summary_logit_bias.param_count():,} params")
        print(f"  Learned log_magnitude: {log_mag:.4f} (magnitude: {math.exp(log_mag):.6f})")
    else:
        print("ERROR: No logit bias found in checkpoint!")
        return

    # ═══════════════════════════════════════════════
    # Alpha sweep
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"ALPHA SWEEP: {len(alphas)} values")
    print(f"{'='*70}")

    results = {}

    for alpha in alphas:
        print(f"\n--- alpha = {alpha:.2f} ---")

        # With summaries
        eval_ds = StreamingEvalDataset(
            dataset_name=config.get("validation_dataset", config["dataset"]),
            split=config.get("validation_split", "train"),
            tokenizer=tokenizer,
            seq_len=config["seq_len"],
            step_token_id=step_token_id,
            step_length=config["step_length"],
            skip_examples=50000,
        )
        loader = DataLoader(eval_ds, batch_size=config.get("batch_size", 2))
        ws_loss, ws_tokens, ws_batches = compute_per_step_with_alpha(
            cct_model, commitment_head, up_project, loader,
            step_token_id, device, args.eval_batches,
            summary_logit_bias=summary_logit_bias, alpha=alpha,
            use_summaries=True,
        )

        # No summaries (alpha=0 equivalent, but we need to compute once)
        if alpha == alphas[0]:
            eval_ds_ns = StreamingEvalDataset(
                dataset_name=config.get("validation_dataset", config["dataset"]),
                split=config.get("validation_split", "train"),
                tokenizer=tokenizer,
                seq_len=config["seq_len"],
                step_token_id=step_token_id,
                step_length=config["step_length"],
                skip_examples=50000,
            )
            loader_ns = DataLoader(eval_ds_ns, batch_size=config.get("batch_size", 2))
            ns_loss, ns_tokens, ns_batches = compute_per_step_with_alpha(
                cct_model, commitment_head, up_project, loader_ns,
                step_token_id, device, args.eval_batches,
                summary_logit_bias=summary_logit_bias, alpha=0.0,
                use_summaries=False,
            )

        # Compute per-step metrics
        alpha_results = {"alpha": alpha, "per_step": {}}
        total_ws_loss = sum(ws_loss.values())
        total_ws_tokens = sum(ws_tokens.values())
        ws_ppl_overall = math.exp(total_ws_loss / total_ws_tokens) if total_ws_tokens > 0 else 0.0
        delta_pct = (ws_ppl_overall - bl_ppl_overall) / bl_ppl_overall * 100

        alpha_results["overall_ppl"] = ws_ppl_overall
        alpha_results["deployment_delta_pct"] = delta_pct

        for si in sorted(ws_loss.keys()):
            ws_ppl = math.exp(ws_loss[si] / ws_tokens[si]) if ws_tokens[si] > 0 else 0.0
            ns_ppl = math.exp(ns_loss[si] / ns_tokens[si]) if si in ns_loss and ns_tokens[si] > 0 else 0.0
            bl_ppl_si = baseline_ppl.get(si, 0.0)
            benefit = ns_ppl - ws_ppl  # positive = summaries help

            alpha_results["per_step"][str(si)] = {
                "baseline_ppl": bl_ppl_si,
                "with_summary_ppl": ws_ppl,
                "no_summary_ppl": ns_ppl,
                "summary_benefit": benefit,
            }
            print(f"  Step {si}: baseline={bl_ppl_si:.2f}  with_sum={ws_ppl:.2f}  no_sum={ns_ppl:.2f}  benefit={benefit:+.2f}")

        print(f"  Overall: PPL={ws_ppl_overall:.2f}  delta={delta_pct:+.2f}%")
        results[f"{alpha:.2f}"] = alpha_results

    # ═══════════════════════════════════════════════
    # Summary: find optimal alpha
    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ALPHA SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline PPL: {bl_ppl_overall:.2f}")
    print(f"  {'Alpha':>6s} | {'PPL':>8s} | {'Delta%':>8s} | {'Benefit@1':>10s} | {'Benefit@2':>10s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10}")

    best_alpha = None
    best_score = float('inf')

    for alpha_key, r in sorted(results.items()):
        ppl = r["overall_ppl"]
        delta = r["deployment_delta_pct"]
        b1 = r["per_step"].get("1", {}).get("summary_benefit", 0.0)
        b2 = r["per_step"].get("2", {}).get("summary_benefit", 0.0)
        print(f"  {float(alpha_key):6.2f} | {ppl:8.2f} | {delta:+8.2f}% | {b1:+10.2f} | {b2:+10.2f}")

        # Optimize: lowest PPL (= best deployment performance)
        if ppl < best_score:
            best_score = ppl
            best_alpha = float(alpha_key)

    print(f"\n  Optimal alpha: {best_alpha:.2f} (PPL={best_score:.2f}, delta={results[f'{best_alpha:.2f}']['deployment_delta_pct']:+.2f}%)")

    # Save results
    results_dir = Path(config.get("results_dir", "./results-tier3-logit-bias-F"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "alpha_sweep_results.json"

    output = {
        "baseline_ppl": bl_ppl_overall,
        "baseline_per_step": {str(k): v for k, v in baseline_ppl.items()},
        "sweep": results,
        "optimal_alpha": best_alpha,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
