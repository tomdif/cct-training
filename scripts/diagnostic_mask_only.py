"""
Diagnostic: CCT mask WITHOUT summary injection.

Isolates how much perplexity cost comes from the mask alone
vs. the summaries actively helping or hurting.

Conditions:
  1. CCT model + CCT mask + NO summaries  (mask-only cost)
  2. CCT model + CCT mask + summaries     (full deployment, for comparison)
  3. CCT model + standard attention        (weights-only, for reference)

If mask-only ≈ deployment: summaries aren't helping or hurting.
If mask-only > deployment: summaries help despite mismatch.
If mask-only < deployment: summaries actively hurt (injection noise).
"""

import argparse
import json
import math
import sys
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
from src.model.cct_attention import build_cct_attention_mask_fast
from scripts.eval_perplexity import StreamingEvalDataset


def compute_perplexity_mask_only(
    model, dataloader, step_token_id, device, max_batches=200,
):
    """CCT attention mask but NO summary injection — pure mask cost."""
    model.eval()
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

            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )

            # Get embeddings (no modification — STEP tokens keep their original embedding)
            if hasattr(model, "get_input_embeddings"):
                embed_layer = model.get_input_embeddings()
            else:
                embed_layer = model.gpt_neox.embed_in

            inputs_embeds = embed_layer(input_ids)

            # Build CCT attention mask (blocks cross-step token attention)
            cct_mask = build_cct_attention_mask_fast(
                seq_len=seq_len,
                boundary_positions=boundary_positions,
                num_prior_summaries=0,
                device=device,
                batch_size=batch_size,
            )

            # Forward with CCT mask, no summary injection
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=cct_mask,
            )
            logits = outputs.logits[:, :-1, :]

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
    parser = argparse.ArgumentParser(description="Diagnostic: mask-only vs deployment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--eval-batches", type=int, default=200)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_token_id = extend_tokenizer(tokenizer)

    batch_size = config.get("batch_size", 4)

    # Load CCT model
    print("Loading CCT model...")
    cct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    cct_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)
    cct_model.load_state_dict(ckpt["model_state_dict"])

    # Load commitment head + up-project for deployment condition
    commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config["d_summary"],
        d_bottleneck=config["d_bottleneck"],
        use_tanh=config.get("use_tanh", True),
        use_l2_norm=config.get("use_l2_norm", True),
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

    # Import the deployment eval function
    from scripts.eval_perplexity import compute_perplexity, compute_perplexity_cct_masked

    # ═══════════════════════════════════════
    # CONDITION 1: CCT mask + NO summaries
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 1: CCT MASK ONLY (no summary injection)")
    print("=" * 60)
    eval_ds1 = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    loader1 = DataLoader(eval_ds1, batch_size=batch_size)
    mask_loss, mask_ppl, mask_tokens, mask_batches = compute_perplexity_mask_only(
        cct_model, loader1, step_token_id, device, args.eval_batches,
    )
    print(f"\n  Mask-only: loss={mask_loss:.4f}  ppl={mask_ppl:.2f}")

    # ═══════════════════════════════════════
    # CONDITION 2: CCT mask + summaries (deployment)
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 2: CCT MASK + SUMMARIES (deployment)")
    print("=" * 60)
    eval_ds2 = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    loader2 = DataLoader(eval_ds2, batch_size=batch_size)
    injection_mode = config.get("summary_injection", "replace")
    deploy_loss, deploy_ppl, deploy_tokens, deploy_batches = compute_perplexity_cct_masked(
        cct_model, commitment_head, up_project, loader2,
        step_token_id, device, args.eval_batches,
        injection_mode=injection_mode,
    )
    print(f"\n  Deployment: loss={deploy_loss:.4f}  ppl={deploy_ppl:.2f}")

    # ═══════════════════════════════════════
    # CONDITION 3: Standard attention (reference)
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 3: STANDARD ATTENTION (weights-only reference)")
    print("=" * 60)
    eval_ds3 = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    loader3 = DataLoader(eval_ds3, batch_size=batch_size)
    std_loss, std_ppl, std_tokens, std_batches = compute_perplexity(
        cct_model, loader3, step_token_id, device, args.eval_batches,
    )
    print(f"\n  Standard attn: loss={std_loss:.4f}  ppl={std_ppl:.2f}")

    # ═══════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════
    # Use previous baseline result for reference
    bl_ppl = 50.04  # from s200 eval (baseline with step_length=200 data)

    delta_mask_pct = (mask_ppl - bl_ppl) / bl_ppl * 100
    delta_deploy_pct = (deploy_ppl - bl_ppl) / bl_ppl * 100
    delta_std_pct = (std_ppl - bl_ppl) / bl_ppl * 100
    summary_effect = deploy_ppl - mask_ppl

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"  Baseline PPL (prior):      {bl_ppl:.2f}")
    print(f"  Standard attn PPL:         {std_ppl:.2f}  (delta: {delta_std_pct:+.2f}%)")
    print(f"  Mask-only PPL:             {mask_ppl:.2f}  (delta: {delta_mask_pct:+.2f}%)")
    print(f"  Deployment PPL:            {deploy_ppl:.2f}  (delta: {delta_deploy_pct:+.2f}%)")
    print(f"")
    print(f"  Summary effect on PPL:     {summary_effect:+.2f}")
    if summary_effect < 0:
        print(f"  -> Summaries HELP ({summary_effect:.2f} PPL reduction)")
    elif summary_effect > 0:
        print(f"  -> Summaries HURT (+{summary_effect:.2f} PPL increase)")
    else:
        print(f"  -> Summaries have NO EFFECT")
    print(f"")
    print(f"  Mask cost (mask-only - std attn): {mask_ppl - std_ppl:+.2f} PPL")

    results = {
        "baseline_ppl_reference": bl_ppl,
        "standard_attention": {"loss": std_loss, "ppl": std_ppl, "delta_pct": delta_std_pct},
        "mask_only": {"loss": mask_loss, "ppl": mask_ppl, "delta_pct": delta_mask_pct},
        "deployment": {"loss": deploy_loss, "ppl": deploy_ppl, "delta_pct": delta_deploy_pct},
        "summary_effect_ppl": summary_effect,
    }
    results_dir = Path(config.get("results_dir", "./results-tier1-e2e"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "diagnostic_mask_only.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
