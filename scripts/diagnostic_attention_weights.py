"""
Diagnostic: Attention weight analysis on summary positions.

For sequential CCT inference, measures how much attention mass
the transformer allocates to prepended summary tokens vs step tokens.

If summary attention is near-zero: model can't route through summaries
(attention routing problem, not information content problem).

If summary attention is significant but PPL doesn't improve:
information content or value projection problem.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
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
from scripts.eval_perplexity import StreamingEvalDataset


def analyze_attention_to_summaries(
    model, commitment_head, up_project, dataloader,
    step_token_id, device, max_batches=50,
):
    """Capture attention weights and measure mass on summary positions."""
    model.eval()
    commitment_head.eval()

    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.gpt_neox.embed_in

    # Collect per-layer, per-head attention to summaries
    # Shape: [n_layers][n_heads] -> list of attention fractions
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    # Per-layer-head: fraction of attention on summary positions
    attn_to_summary = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    # Per-layer-head: fraction on first summary specifically
    attn_to_first_summary = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    # Per-layer: average across heads
    attn_to_summary_by_layer = [[] for _ in range(n_layers)]

    n_batches = 0
    n_steps_with_summaries = 0

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

                n_summaries = len(committed_summaries)
                if n_summaries > 0:
                    summary_embeds_list = []
                    summary_pos_list = []
                    for s_idx in range(n_summaries):
                        up_proj = up_project(
                            committed_summaries[s_idx]
                        ).unsqueeze(1).to(step_embeds.dtype)
                        summary_embeds_list.append(up_proj)
                        summary_pos_list.append(boundary_positions[s_idx])

                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    inputs_embeds = torch.cat(
                        [summary_embeds, step_embeds], dim=1
                    )
                    summary_positions = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat(
                        [summary_positions, step_positions], dim=1
                    )
                else:
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                # Forward with attention output
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                )

                # Commitment summary for non-last steps
                is_last_step = (step_idx == len(step_ranges) - 1)
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][:, n_summaries:, :]
                    summary = commitment_head(step_hidden.float())
                    committed_summaries.append(summary)

                # Analyze attention weights (only for steps with summaries)
                if n_summaries > 0:
                    n_steps_with_summaries += 1
                    # outputs.attentions is tuple of (B, n_heads, seq_len, seq_len)
                    total_len = n_summaries + step_len

                    for layer_idx, attn_weights in enumerate(outputs.attentions):
                        # attn_weights: (B, n_heads, total_len, total_len)
                        # We care about: for step tokens (positions n_summaries:),
                        # how much attention goes to summary positions (0:n_summaries)?

                        # Step token queries attending to all keys
                        step_query_attn = attn_weights[:, :, n_summaries:, :]
                        # (B, n_heads, step_len, total_len)

                        # Attention mass on summary keys
                        attn_on_summaries = step_query_attn[:, :, :, :n_summaries]
                        # (B, n_heads, step_len, n_summaries)

                        # Mean attention fraction per head (averaged over queries and batch)
                        summary_frac = attn_on_summaries.sum(dim=-1).mean(dim=(0, 2))
                        # (n_heads,) — average fraction of attention going to summaries

                        for head_idx in range(n_heads):
                            frac = summary_frac[head_idx].item()
                            attn_to_summary[layer_idx][head_idx].append(frac)

                        # Layer average
                        layer_avg = summary_frac.mean().item()
                        attn_to_summary_by_layer[layer_idx].append(layer_avg)

                        # Attention to first summary specifically
                        if n_summaries >= 1:
                            attn_first = step_query_attn[:, :, :, 0].mean(dim=(0, 2))
                            for head_idx in range(n_heads):
                                attn_to_first_summary[layer_idx][head_idx].append(
                                    attn_first[head_idx].item()
                                )

            n_batches += 1
            if n_batches % 10 == 0:
                print(f"  Processed {n_batches}/{max_batches} batches, "
                      f"{n_steps_with_summaries} steps with summaries")

    return {
        "attn_to_summary": attn_to_summary,
        "attn_to_first_summary": attn_to_first_summary,
        "attn_to_summary_by_layer": attn_to_summary_by_layer,
        "n_batches": n_batches,
        "n_steps_with_summaries": n_steps_with_summaries,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnostic: attention on summaries")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--eval-batches", type=int, default=50)
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
        torch_dtype=torch.float32,  # float32 for attention weight precision
    ).to(device)
    cct_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)
    cct_model.load_state_dict(ckpt["model_state_dict"])

    # Load commitment head + up-project
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

    # Build eval dataset
    eval_dataset = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    print(f"\nAnalyzing attention patterns ({args.eval_batches} batches)...")
    print(f"  step_length={config['step_length']}, seq_len={config['seq_len']}")
    print(f"  Expected ~{config['seq_len'] // config['step_length'] - 1} steps with summaries per sequence\n")

    results = analyze_attention_to_summaries(
        cct_model, commitment_head, up_project, eval_loader,
        step_token_id, device, args.eval_batches,
    )

    n_layers = results["n_layers"]
    n_heads = results["n_heads"]

    # Print layer-by-layer summary
    print("\n" + "=" * 70)
    print("ATTENTION TO SUMMARY TOKENS (fraction of total attention mass)")
    print("=" * 70)
    print(f"\nSampled {results['n_steps_with_summaries']} steps with summaries "
          f"across {results['n_batches']} batches\n")

    # Uniform baseline: if seq has n_sum summaries and step_len tokens,
    # uniform attention to summaries = n_sum / (n_sum + step_len)
    step_len = config["step_length"]
    n_sum_typical = 1  # most steps have 1 prior summary at s=200
    uniform_baseline = n_sum_typical / (n_sum_typical + step_len)
    print(f"Uniform attention baseline (1 summary, {step_len} step tokens): "
          f"{uniform_baseline:.4f} ({uniform_baseline*100:.2f}%)\n")

    print(f"{'Layer':>6} | {'Mean':>8} | {'Min Head':>10} | {'Max Head':>10} | {'Head Details'}")
    print("-" * 70)

    layer_means = []
    for layer_idx in range(n_layers):
        layer_data = results["attn_to_summary_by_layer"][layer_idx]
        if not layer_data:
            continue
        layer_mean = np.mean(layer_data)
        layer_means.append(layer_mean)

        head_means = []
        for head_idx in range(n_heads):
            hdata = results["attn_to_summary"][layer_idx][head_idx]
            if hdata:
                head_means.append(np.mean(hdata))
            else:
                head_means.append(0.0)

        min_head = min(head_means)
        max_head = max(head_means)
        max_head_idx = head_means.index(max_head)

        head_str = " ".join(f"{h:.3f}" for h in head_means)
        print(f"  L{layer_idx:>3} | {layer_mean:>7.4f} | {min_head:>9.4f} | "
              f"{max_head:>9.4f} (H{max_head_idx}) | {head_str}")

    overall_mean = np.mean(layer_means) if layer_means else 0
    print("-" * 70)
    print(f"  Overall mean: {overall_mean:.4f} ({overall_mean*100:.2f}%)")
    print(f"  Uniform baseline: {uniform_baseline:.4f} ({uniform_baseline*100:.2f}%)")
    ratio = overall_mean / uniform_baseline if uniform_baseline > 0 else 0
    print(f"  Ratio (actual/uniform): {ratio:.2f}x")

    if overall_mean < uniform_baseline * 0.5:
        print("\n  DIAGNOSIS: Summary tokens receive BELOW-UNIFORM attention.")
        print("  -> Attention routing problem confirmed.")
        print("  -> Model actively avoids summary positions.")
    elif overall_mean < uniform_baseline * 1.5:
        print("\n  DIAGNOSIS: Summary tokens receive NEAR-UNIFORM attention.")
        print("  -> Model treats summaries like any other token.")
        print("  -> Summaries not differentiated; value content may be noise.")
    else:
        print("\n  DIAGNOSIS: Summary tokens receive ABOVE-UNIFORM attention.")
        print("  -> Model routes attention to summaries.")
        print("  -> If PPL doesn't improve, problem is in value/information content.")

    # Find "summary specialist" heads (>3x uniform)
    print("\n\nSUMMARY SPECIALIST HEADS (>3x uniform baseline):")
    print("-" * 50)
    specialists = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            hdata = results["attn_to_summary"][layer_idx][head_idx]
            if hdata:
                hmean = np.mean(hdata)
                if hmean > uniform_baseline * 3:
                    specialists.append((layer_idx, head_idx, hmean))
                    print(f"  Layer {layer_idx}, Head {head_idx}: "
                          f"{hmean:.4f} ({hmean/uniform_baseline:.1f}x uniform)")
    if not specialists:
        print("  None found. No heads specialize in reading summaries.")

    # Save detailed results
    results_dir = Path(config.get("results_dir", "./results-tier1-seq-s200"))
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "config": {
            "step_length": config["step_length"],
            "seq_len": config["seq_len"],
            "d_summary": config["d_summary"],
        },
        "uniform_baseline": uniform_baseline,
        "overall_mean_attn_to_summary": overall_mean,
        "ratio_to_uniform": ratio,
        "n_steps_with_summaries": results["n_steps_with_summaries"],
        "n_batches": results["n_batches"],
        "layer_means": [float(x) for x in layer_means],
        "head_means": [
            [float(np.mean(results["attn_to_summary"][l][h]))
             if results["attn_to_summary"][l][h] else 0.0
             for h in range(n_heads)]
            for l in range(n_layers)
        ],
        "specialists": [
            {"layer": l, "head": h, "attn_frac": float(f)}
            for l, h, f in specialists
        ],
    }

    out_path = results_dir / "diagnostic_attention_weights.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
