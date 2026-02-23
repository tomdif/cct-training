"""
Sufficiency Probe: THE critical measurement.

Method:
  1. Run CCT-trained model on held-out data with step boundaries.
  2. At each step boundary, extract:
     a. The commitment summary (d_summary vector)
     b. The full hidden state of the step (d_model vector, mean-pooled)
  3. Train a LINEAR probe: summary -> hidden state
  4. Measure R² reconstruction error on held-out pairs.

If R² > 0.85 (Tier 1) / 0.90 (Tier 2), the summary IS a sufficient statistic.

See CCT_IMPLEMENTATION_GUIDE.md Section 2.4.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model.commitment_head import CommitmentHead
from src.training.data_pipeline import get_step_boundary_positions, get_step_ranges


def collect_summary_hidden_pairs(
    model: nn.Module,
    commitment_head: CommitmentHead,
    dataloader,
    step_token_id: int,
    device: torch.device,
    max_pairs: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (summary, step_hidden_mean) pairs from held-out data.

    Args:
        model: The transformer model.
        commitment_head: The trained commitment head.
        dataloader: Held-out data loader yielding input_ids tensors.
        step_token_id: ID of the STEP token.
        device: Torch device.
        max_pairs: Maximum number of pairs to collect.

    Returns:
        summaries: (N, d_summary)
        hidden_means: (N, d_model)
    """
    model.eval()
    commitment_head.eval()

    all_summaries = []
    all_hiddens = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            else:
                input_ids = batch.to(device)

            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (B, S, D)

            # Process each item in the batch
            for b in range(input_ids.shape[0]):
                boundaries = get_step_boundary_positions(
                    input_ids[b].tolist(), step_token_id
                )
                step_ranges = get_step_ranges(input_ids.shape[1], boundaries)

                for start, end in step_ranges:
                    if end - start < 2:
                        continue
                    step_hidden = hidden_states[b, start:end, :]  # (step_len, D)
                    step_mean = step_hidden.float().mean(dim=0)  # (D,)

                    summary = commitment_head(
                        step_hidden.float().unsqueeze(0)
                    ).squeeze(0)  # (d_summary,)

                    all_summaries.append(summary.cpu())
                    all_hiddens.append(step_mean.cpu())

                    if len(all_summaries) >= max_pairs:
                        return torch.stack(all_summaries), torch.stack(all_hiddens)

    if not all_summaries:
        raise ValueError("No summary-hidden pairs collected. Check data and step_token_id.")

    summaries = torch.stack(all_summaries)
    hiddens = torch.stack(all_hiddens)

    # Filter out any NaN pairs
    valid = ~(summaries.isnan().any(dim=1) | hiddens.isnan().any(dim=1))
    if valid.sum() < summaries.shape[0]:
        n_bad = summaries.shape[0] - valid.sum().item()
        print(f"  Warning: filtered {n_bad} NaN pairs out of {summaries.shape[0]}")
        summaries = summaries[valid]
        hiddens = hiddens[valid]

    return summaries, hiddens


def train_linear_probe(
    summaries: torch.Tensor,
    hidden_means: torch.Tensor,
    train_fraction: float = 0.8,
    lr: float = 1e-3,
    epochs: int = 100,
) -> tuple[nn.Linear, float]:
    """Train a linear probe and measure R².

    Args:
        summaries: (N, d_summary) — input features.
        hidden_means: (N, d_model) — targets.
        train_fraction: Fraction of data for training.
        lr: Learning rate.
        epochs: Training epochs.

    Returns:
        probe: Trained linear probe.
        r2: R² score on held-out data.
    """
    N = summaries.shape[0]
    d_summary = summaries.shape[1]
    d_model = hidden_means.shape[1]

    # Split train/val
    split = int(N * train_fraction)
    perm = torch.randperm(N)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, Y_train = summaries[train_idx], hidden_means[train_idx]
    X_val, Y_val = summaries[val_idx], hidden_means[val_idx]

    # Train linear probe
    probe = nn.Linear(d_summary, d_model)
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    best_r2 = -float("inf")
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        probe.train()
        pred = probe(X_train)
        loss = nn.functional.mse_loss(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate — use total R² (not per-dimension average, which is
        # dominated by low-variance dimensions)
        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val).numpy()
            val_true = Y_val.numpy()
            ss_res = ((val_true - val_pred) ** 2).sum()
            ss_tot = ((val_true - val_true.mean(axis=0)) ** 2).sum()
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        if r2 > best_r2:
            best_r2 = r2
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Diagnostics
    print(f"  Probe training: {epoch+1} epochs, best R²={best_r2:.4f}")
    print(f"  Target stats: mean={Y_val.mean():.4f} std={Y_val.std():.4f}")
    print(f"  Summary stats: mean={X_val.mean():.4f} std={X_val.std():.4f} (L2-normed)")
    with torch.no_grad():
        final_pred = probe(X_val).numpy()
        final_mse = ((val_true - final_pred) ** 2).mean()
        target_var = val_true.var()
        print(f"  Val MSE={final_mse:.4f}, target var={target_var:.4f}")
        print(f"  Per-dim MSE={final_mse:.6f}, per-dim var={target_var:.6f}")

    return probe, best_r2


def evaluate_sufficiency(
    model: nn.Module,
    commitment_head: CommitmentHead,
    dataloader,
    step_token_id: int,
    device: torch.device,
    max_pairs: int = 5000,
) -> dict:
    """Full sufficiency evaluation pipeline.

    Returns dict with R² and other diagnostics.
    """
    summaries, hidden_means = collect_summary_hidden_pairs(
        model, commitment_head, dataloader, step_token_id, device, max_pairs
    )

    probe, r2 = train_linear_probe(summaries, hidden_means)

    return {
        "sufficiency_r2": r2,
        "n_pairs": summaries.shape[0],
        "d_summary": summaries.shape[1],
        "d_model": hidden_means.shape[1],
        "compression_factor": hidden_means.shape[1] / summaries.shape[1],
    }
