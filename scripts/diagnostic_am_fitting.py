"""
AM Value Fitting Diagnostic -- F19 v3 Claim 12 Test

Background:
  - Attention routing works: 30 specialist heads, 3.2x uniform attention to summaries
  - But summaries only recover 0.63 PPL (effectively nothing)
  - L_suf=0.32: commitment head encodes well, linear probe can decode
  - Hypothesis: up-projection into V-space is the broken link

Three eval conditions:
  A. Current: trained up_project MLP (128->1024->768)
  B. Oracle: inject mean embedding of prior-step tokens (ceiling, uses original tokens)
  C. AM-fitted: least-squares linear fit from summary -> embedding space

Interpretation:
  - If B << A: V-content is the bottleneck
  - If C ~ B: AM linear fit captures what's needed
  - If C << A: AM post-projection optimization works (validates F19 claim 12)

Why mean embedding is the right target:
  - V-projection is linear: V(mean(x_i)) = mean(V(x_i))
  - So mean embedding at input level produces mean V-content at every layer
  - This is the best single-vector approximation of N prior tokens
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
    get_step_boundary_positions,
    get_step_ranges,
)
from src.model.commitment_head import CommitmentHead
from src.model.summary_buffer import SummaryBuffer
from scripts.eval_perplexity import StreamingEvalDataset


def collect_am_fitting_data(
    model, commitment_head, embed_layer, up_project,
    dataloader, step_token_id, device, max_batches=20,
):
    """Collect (summary, target_mean_embed) pairs for AM fitting.

    Runs sequential forward. For each non-last step, records:
      - The 128-d commitment summary
      - The mean input embedding of that step's tokens (768-d)
    """
    all_summaries = []
    all_targets = []
    all_up_outputs = []  # what current up_project produces, for comparison
    n_batches = 0

    model.eval()
    commitment_head.eval()

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

                n_sums = len(committed_summaries)
                if n_sums > 0:
                    s_list = []
                    s_pos = []
                    for si in range(n_sums):
                        up = up_project(
                            committed_summaries[si]
                        ).unsqueeze(1).to(step_embeds.dtype)
                        s_list.append(up)
                        s_pos.append(boundary_positions[si])
                    sum_embeds = torch.cat(s_list, dim=1)
                    inputs_embeds = torch.cat([sum_embeds, step_embeds], dim=1)
                    spos = torch.tensor(
                        s_pos, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([spos, step_positions], dim=1)
                else:
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )

                is_last = (step_idx == len(step_ranges) - 1)
                if not is_last:
                    step_hidden = outputs.hidden_states[-1][:, n_sums:, :]
                    summary = commitment_head(step_hidden.float())
                    committed_summaries.append(summary)

                    # Record: summary, up_project output, target mean embedding
                    mean_embed = step_embeds.float().mean(dim=1)  # (B, d_model)
                    up_out = up_project(summary).float()  # (B, d_model)

                    all_summaries.append(summary.detach().cpu())
                    all_targets.append(mean_embed.detach().cpu())
                    all_up_outputs.append(up_out.detach().cpu())

            n_batches += 1
            if n_batches % 5 == 0:
                print(f"  Fitting data: {n_batches}/{max_batches} batches")

    S = torch.cat(all_summaries, dim=0)   # (N, d_summary)
    T = torch.cat(all_targets, dim=0)     # (N, d_model)
    U = torch.cat(all_up_outputs, dim=0)  # (N, d_model)
    print(f"  Collected {S.shape[0]} fitting samples")
    return S, T, U


def sequential_eval_am(
    model, commitment_head, embed_layer, up_project,
    dataloader, step_token_id, device,
    mode="current", C_v=None, max_batches=200,
):
    """Sequential eval with configurable summary embedding mode.

    Modes:
      "current": use trained up_project MLP
      "oracle": use mean embedding of prior-step tokens
      "am_fitted": use summary @ C_v (linear fit)
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

            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(seq_len, boundary_positions)

            committed_summaries = []
            oracle_embeds = []

            for step_idx, (start, end) in enumerate(step_ranges):
                step_len = end - start
                if step_len <= 0:
                    continue
                step_token_ids = input_ids[:, start:end]
                step_embeds = embed_layer(step_token_ids)

                step_positions = torch.arange(
                    start, end, device=device
                ).unsqueeze(0).expand(batch_size, -1)

                n_sums = len(committed_summaries)
                if n_sums > 0:
                    s_list = []
                    s_pos = []
                    for si in range(n_sums):
                        if mode == "current":
                            emb = up_project(committed_summaries[si])
                        elif mode == "oracle":
                            emb = oracle_embeds[si].to(device)
                        elif mode == "am_fitted":
                            emb = (committed_summaries[si].float() @ C_v.to(device))
                        else:
                            raise ValueError(f"Unknown mode: {mode}")
                        s_list.append(emb.unsqueeze(1).to(step_embeds.dtype))
                        s_pos.append(boundary_positions[si])

                    sum_embeds = torch.cat(s_list, dim=1)
                    inputs_embeds = torch.cat([sum_embeds, step_embeds], dim=1)
                    spos = torch.tensor(
                        s_pos, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([spos, step_positions], dim=1)
                else:
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )

                is_last = (step_idx == len(step_ranges) - 1)
                if not is_last:
                    step_hidden = outputs.hidden_states[-1][:, n_sums:, :]
                    summary = commitment_head(step_hidden.float())
                    committed_summaries.append(summary)
                    # Always compute oracle embed (cheap, needed for oracle mode)
                    oracle_embeds.append(step_embeds.float().mean(dim=1).detach())

                # Loss computation
                step_logits = outputs.logits[
                    :, n_sums:n_sums + step_len - 1, :
                ]
                step_labels = step_token_ids[:, 1:].clone()
                step_labels[step_labels == step_token_id] = -100

                if n_sums > 0:
                    cross_logit = outputs.logits[
                        :, n_sums - 1:n_sums, :
                    ]
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
                total_loss += loss.item()
                total_tokens += n_valid

            n_batches += 1
            if n_batches % 50 == 0:
                avg = total_loss / total_tokens
                print(f"  [{mode}] batch {n_batches}/{max_batches} | "
                      f"avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens, n_batches


def main():
    parser = argparse.ArgumentParser(
        description="AM Value Fitting Diagnostic"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--fit-batches", type=int, default=30)
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

    # Load model
    print("Loading CCT model...")
    cct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    cct_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)
    cct_model.load_state_dict(ckpt["model_state_dict"])

    if hasattr(cct_model, "get_input_embeddings"):
        embed_layer = cct_model.get_input_embeddings()
    else:
        embed_layer = cct_model.gpt_neox.embed_in

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

    # ================================================================
    # PHASE 1: Collect fitting data
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Collecting AM fitting data")
    print("=" * 60)
    fit_dataset = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=40000,  # different skip from eval (50000) to avoid overlap
    )
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size)

    S, T, U = collect_am_fitting_data(
        cct_model, commitment_head, embed_layer, up_project,
        fit_loader, step_token_id, device, args.fit_batches,
    )

    # ================================================================
    # PHASE 2: Fit AM C_v via least squares
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fitting AM value projection (least squares)")
    print("=" * 60)

    # Solve: S @ C_v = T  (minimize ||S @ C_v - T||^2)
    # S: (N, 128), T: (N, 768) -> C_v: (128, 768)
    result = torch.linalg.lstsq(S, T)
    C_v = result.solution  # (d_summary, d_model)

    # Evaluate fit quality
    am_fitted = S @ C_v
    mse_am = ((am_fitted - T) ** 2).mean().item()
    ss_res = ((am_fitted - T) ** 2).sum().item()
    ss_tot = ((T - T.mean(dim=0)) ** 2).sum().item()
    r2_am = 1 - ss_res / ss_tot

    # Compare current up_project
    mse_current = ((U - T) ** 2).mean().item()
    ss_res_curr = ((U - T) ** 2).sum().item()
    r2_current = 1 - ss_res_curr / ss_tot

    # Cosine similarity between up_project output and target
    cos_current = torch.nn.functional.cosine_similarity(U, T, dim=1).mean().item()
    cos_am = torch.nn.functional.cosine_similarity(am_fitted, T, dim=1).mean().item()

    print(f"\n  Target: mean embedding of step tokens (768-d)")
    print(f"  Fitting samples: {S.shape[0]}")
    print(f"")
    print(f"  {'Metric':<25} {'Current up_project':>20} {'AM-fitted (lstsq)':>20}")
    print(f"  {'-'*65}")
    print(f"  {'MSE to target':<25} {mse_current:>20.4f} {mse_am:>20.4f}")
    print(f"  {'R^2':<25} {r2_current:>20.4f} {r2_am:>20.4f}")
    print(f"  {'Cosine similarity':<25} {cos_current:>20.4f} {cos_am:>20.4f}")
    print(f"")

    if mse_am < mse_current:
        improvement = (mse_current - mse_am) / mse_current * 100
        print(f"  AM fit is {improvement:.1f}% closer to target than trained up_project")
    else:
        print(f"  Trained up_project is closer to target (AM fit not better)")

    # ================================================================
    # PHASE 3: Evaluate three conditions
    # ================================================================

    # --- Condition A: Current up_project ---
    print("\n" + "=" * 60)
    print("CONDITION A: Sequential with CURRENT up_project (trained MLP)")
    print("=" * 60)
    eval_ds_a = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    loader_a = DataLoader(eval_ds_a, batch_size=batch_size)
    loss_a, ppl_a, tok_a, bat_a = sequential_eval_am(
        cct_model, commitment_head, embed_layer, up_project,
        loader_a, step_token_id, device,
        mode="current", max_batches=args.eval_batches,
    )
    print(f"\n  Current up_project: loss={loss_a:.4f}  ppl={ppl_a:.2f}")

    # --- Condition B: Oracle (mean embedding) ---
    print("\n" + "=" * 60)
    print("CONDITION B: Sequential with ORACLE (mean prior-step embedding)")
    print("=" * 60)
    eval_ds_b = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    loader_b = DataLoader(eval_ds_b, batch_size=batch_size)
    loss_b, ppl_b, tok_b, bat_b = sequential_eval_am(
        cct_model, commitment_head, embed_layer, up_project,
        loader_b, step_token_id, device,
        mode="oracle", max_batches=args.eval_batches,
    )
    print(f"\n  Oracle: loss={loss_b:.4f}  ppl={ppl_b:.2f}")

    # --- Condition C: AM-fitted ---
    print("\n" + "=" * 60)
    print("CONDITION C: Sequential with AM-FITTED C_v (least squares)")
    print("=" * 60)
    eval_ds_c = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    loader_c = DataLoader(eval_ds_c, batch_size=batch_size)
    loss_c, ppl_c, tok_c, bat_c = sequential_eval_am(
        cct_model, commitment_head, embed_layer, up_project,
        loader_c, step_token_id, device,
        mode="am_fitted", C_v=C_v, max_batches=args.eval_batches,
    )
    print(f"\n  AM-fitted: loss={loss_c:.4f}  ppl={ppl_c:.2f}")

    # ================================================================
    # COMPARISON
    # ================================================================
    baseline_ppl = 50.04  # from s200 eval

    delta_a = (ppl_a - baseline_ppl) / baseline_ppl * 100
    delta_b = (ppl_b - baseline_ppl) / baseline_ppl * 100
    delta_c = (ppl_c - baseline_ppl) / baseline_ppl * 100

    print("\n" + "=" * 60)
    print("AM VALUE FITTING RESULTS")
    print("=" * 60)
    print(f"  Baseline PPL:               {baseline_ppl:.2f}")
    print(f"  A. Current up_project PPL:  {ppl_a:.2f}  (delta: {delta_a:+.2f}%)")
    print(f"  B. Oracle (mean embed) PPL: {ppl_b:.2f}  (delta: {delta_b:+.2f}%)")
    print(f"  C. AM-fitted C_v PPL:       {ppl_c:.2f}  (delta: {delta_c:+.2f}%)")
    print(f"")
    print(f"  Oracle improvement over current:    {ppl_a - ppl_b:+.2f} PPL")
    print(f"  AM-fitted improvement over current: {ppl_a - ppl_c:+.2f} PPL")
    print(f"  AM-fitted vs oracle gap:            {ppl_c - ppl_b:+.2f} PPL")
    print(f"")

    # Interpretation
    oracle_improvement = ppl_a - ppl_b
    am_improvement = ppl_a - ppl_c

    if oracle_improvement > 2.0:
        print(f"  FINDING: Oracle improves by {oracle_improvement:.1f} PPL")
        print(f"  -> V-content IS the bottleneck. Better embeddings help.")
        if am_improvement > oracle_improvement * 0.5:
            print(f"  -> AM-fitted captures {am_improvement/oracle_improvement*100:.0f}% "
                  f"of oracle improvement")
            print(f"  -> F19 claim 12 VALIDATED: eval-time AM fitting works")
        else:
            print(f"  -> AM linear fit captures only "
                  f"{am_improvement/oracle_improvement*100:.0f}% of oracle")
            print(f"  -> Linear mapping insufficient; nonlinear fit may be needed")
    elif oracle_improvement > 0.5:
        print(f"  FINDING: Oracle improves by {oracle_improvement:.1f} PPL (modest)")
        print(f"  -> V-content partially contributes but mask cost dominates")
    else:
        print(f"  FINDING: Oracle does NOT improve PPL ({oracle_improvement:.1f})")
        print(f"  -> Problem is NOT V-content. The mask cost is fundamental.")
        print(f"  -> Single summary token cannot replace 200 tokens of context")

    # Save results
    results = {
        "fitting": {
            "n_samples": int(S.shape[0]),
            "mse_current_up_project": mse_current,
            "mse_am_fitted": mse_am,
            "r2_current": r2_current,
            "r2_am": r2_am,
            "cosine_current": cos_current,
            "cosine_am": cos_am,
        },
        "eval": {
            "baseline_ppl": baseline_ppl,
            "current": {"loss": loss_a, "ppl": ppl_a, "delta_pct": delta_a},
            "oracle": {"loss": loss_b, "ppl": ppl_b, "delta_pct": delta_b},
            "am_fitted": {"loss": loss_c, "ppl": ppl_c, "delta_pct": delta_c},
        },
        "improvements": {
            "oracle_over_current_ppl": float(oracle_improvement),
            "am_over_current_ppl": float(am_improvement),
            "am_vs_oracle_gap_ppl": float(ppl_c - ppl_b),
        },
        "config": {
            "step_length": config["step_length"],
            "d_summary": config["d_summary"],
            "d_model": config["d_model"],
        },
    }

    results_dir = Path(config.get("results_dir", "./results-tier1-seq-s200"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "diagnostic_am_fitting.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save C_v for potential reuse
    cv_path = results_dir / "am_fitted_Cv.pt"
    torch.save(C_v, cv_path)
    print(f"Saved C_v matrix to {cv_path}")


if __name__ == "__main__":
    main()
