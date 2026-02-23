"""
Perplexity comparison: CCT model vs baseline on the same held-out data.

Evaluates THREE conditions:
  1. Baseline model (standard attention)
  2. CCT model with standard attention (weights-only test)
  3. CCT model with CCT attention mask + summary injection (deployment scenario)

Usage:
  python scripts/eval_perplexity.py --config configs/tier2_410m.yaml \
    --cct-checkpoint ./checkpoints-tier2/cct-final.pt \
    --baseline-dir ./checkpoints-tier2/baseline-final \
    --eval-batches 200
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
    annotate_step_boundaries,
    get_step_boundary_positions,
    get_step_ranges,
)
from src.model.commitment_head import CommitmentHead
from src.model.summary_buffer import SummaryBuffer
from src.model.cct_attention import build_cct_attention_mask_fast


class StreamingEvalDataset(torch.utils.data.IterableDataset):
    """Deterministic eval data — skips first `skip_examples` to avoid train overlap."""

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


def compute_perplexity(model, dataloader, step_token_id, device, max_batches=200):
    """Compute perplexity with standard causal attention. Masks STEP tokens from loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            labels = input_ids.clone()
            labels[labels == step_token_id] = -100
            labels = labels[:, 1:]

            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (labels.reshape(-1) != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid
            n_batches += 1

            if n_batches % 50 == 0:
                avg = total_loss / total_tokens
                print(f"  batch {n_batches}/{max_batches} | avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens, n_batches


def compute_perplexity_cct_masked(
    model, commitment_head, up_project, dataloader,
    step_token_id, device, max_batches=200,
):
    """Compute perplexity with CCT attention mask and summary injection.

    This replicates the actual CCT deployment scenario:
    1. Pass 1: standard forward to get hidden states
    2. Compute commitment summaries at step boundaries
    3. Pass 2: forward with CCT mask + injected summaries
    4. Measure loss on Pass 2 logits
    """
    model.eval()
    commitment_head.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            # Find step boundaries (use first batch element — uniform annotation)
            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(seq_len, boundary_positions)

            # === Pass 1: standard forward to get hidden states ===
            outputs_pass1 = model(input_ids, output_hidden_states=True)
            hidden_states = outputs_pass1.hidden_states[-1]  # (B, S, D)

            # Compute commitment summaries at each step boundary
            summaries = []
            for start, end in step_ranges[:-1]:  # all steps except last
                step_hidden = hidden_states[:, start:end, :]  # (B, step_len, D)
                step_hidden_f32 = step_hidden.float()
                summary = commitment_head(step_hidden_f32)  # (B, d_summary)
                summaries.append(summary)

            # === Pass 2: CCT-masked forward ===
            if hasattr(model, "get_input_embeddings"):
                embed_layer = model.get_input_embeddings()
            else:
                embed_layer = model.gpt_neox.embed_in

            inputs_embeds = embed_layer(input_ids)  # (B, S, D)

            # Inject up-projected summaries at STEP token positions
            if summaries:
                for idx, bp in enumerate(boundary_positions):
                    if idx < len(summaries):
                        up_proj = up_project(summaries[idx])  # (B, D)
                        inputs_embeds[:, bp, :] = up_proj.to(inputs_embeds.dtype)

            # Build CCT attention mask
            cct_mask = build_cct_attention_mask_fast(
                seq_len=seq_len,
                boundary_positions=boundary_positions,
                num_prior_summaries=0,
                device=device,
                batch_size=batch_size,
            )

            # Forward with CCT mask
            outputs_pass2 = model(
                inputs_embeds=inputs_embeds,
                attention_mask=cct_mask,
            )
            logits = outputs_pass2.logits[:, :-1, :]

            # Labels: mask STEP tokens
            labels = input_ids.clone()
            labels[labels == step_token_id] = -100
            labels = labels[:, 1:]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (labels.reshape(-1) != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid
            n_batches += 1

            if n_batches % 50 == 0:
                avg = total_loss / total_tokens
                print(f"  batch {n_batches}/{max_batches} | avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens, n_batches


def main():
    parser = argparse.ArgumentParser(description="CCT vs Baseline Perplexity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--eval-batches", type=int, default=200)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) backend")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found, evaluation on CPU (slow)")
    model_name = config["base_model"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_token_id = extend_tokenizer(tokenizer)

    # Build eval dataset (same for all conditions)
    eval_dataset = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )

    batch_size = config.get("batch_size", 4)

    # ══════════════════════════════════════════════════════════
    # CONDITION 1: BASELINE (standard attention)
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("CONDITION 1: BASELINE (standard attention)")
    print("=" * 60)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_dir,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    bl_loss, bl_ppl, bl_tokens, bl_batches = compute_perplexity(
        baseline_model, eval_loader, step_token_id, device, args.eval_batches
    )
    print(f"\n  Baseline: loss={bl_loss:.4f}  ppl={bl_ppl:.2f}  ({bl_tokens} tokens, {bl_batches} batches)")

    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # CONDITION 2: CCT (standard attention — weights-only test)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 2: CCT (standard attention — weights-only)")
    print("=" * 60)
    cct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    cct_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)
    cct_model.load_state_dict(ckpt["model_state_dict"])

    eval_loader2 = DataLoader(eval_dataset, batch_size=batch_size)
    cct_std_loss, cct_std_ppl, cct_std_tokens, cct_std_batches = compute_perplexity(
        cct_model, eval_loader2, step_token_id, device, args.eval_batches
    )
    print(f"\n  CCT (std attn): loss={cct_std_loss:.4f}  ppl={cct_std_ppl:.2f}  ({cct_std_tokens} tokens, {cct_std_batches} batches)")

    # ══════════════════════════════════════════════════════════
    # CONDITION 3: CCT (CCT attention mask — deployment scenario)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 3: CCT (CCT mask + summaries — deployment)")
    print("=" * 60)

    # Load commitment head and up-projection (match training config)
    commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config["d_summary"],
        d_bottleneck=config["d_bottleneck"],
        use_tanh=config.get("use_tanh", True),
        use_l2_norm=config.get("use_l2_norm", True),
        noise_injection=config.get("noise_injection", False),
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

    eval_loader3 = DataLoader(eval_dataset, batch_size=batch_size)
    cct_mask_loss, cct_mask_ppl, cct_mask_tokens, cct_mask_batches = compute_perplexity_cct_masked(
        cct_model, commitment_head, up_project, eval_loader3,
        step_token_id, device, args.eval_batches,
    )
    print(f"\n  CCT (masked): loss={cct_mask_loss:.4f}  ppl={cct_mask_ppl:.2f}  ({cct_mask_tokens} tokens, {cct_mask_batches} batches)")

    del cct_model, commitment_head, up_project, summary_buffer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    delta_std_pct = (cct_std_ppl - bl_ppl) / bl_ppl * 100
    delta_mask_pct = (cct_mask_ppl - bl_ppl) / bl_ppl * 100

    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON")
    print("=" * 60)
    print(f"  Baseline PPL:             {bl_ppl:.2f}")
    print(f"  CCT (std attn) PPL:       {cct_std_ppl:.2f}  (delta: {delta_std_pct:+.2f}%)")
    print(f"  CCT (masked+summary) PPL: {cct_mask_ppl:.2f}  (delta: {delta_mask_pct:+.2f}%)")
    print(f"  Target:                   < +10%")
    print(f"  Weights-only result:      {'PASS' if delta_std_pct < 10 else 'FAIL'}")
    print(f"  Deployment result:        {'PASS' if delta_mask_pct < 10 else 'FAIL'}")

    # Save results
    results = {
        "baseline": {
            "loss": bl_loss, "perplexity": bl_ppl,
            "tokens": bl_tokens, "batches": bl_batches,
        },
        "cct_standard_attention": {
            "loss": cct_std_loss, "perplexity": cct_std_ppl,
            "tokens": cct_std_tokens, "batches": cct_std_batches,
            "delta_pct": delta_std_pct,
            "pass": delta_std_pct < 10,
        },
        "cct_masked_deployment": {
            "loss": cct_mask_loss, "perplexity": cct_mask_ppl,
            "tokens": cct_mask_tokens, "batches": cct_mask_batches,
            "delta_pct": delta_mask_pct,
            "pass": delta_mask_pct < 10,
        },
    }
    results_dir = Path(config.get("results_dir", "./results-tier2"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "perplexity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
