"""
Irreversibility-Aware Loss Function

Four components, weighted by curriculum:

1. alpha * L_standard:   Standard next-token prediction (cross-entropy)
2. beta  * L_validity:   Per-step validity — each step correct given ONLY prior summaries
3. gamma * L_conclusion: Conclusion from premises — final answer from summaries alone
4. delta * L_sufficiency: Can a linear probe recover step info from summary?

The key insight: L_validity and L_conclusion FORCE sufficiency.
If the model can only use summaries (not raw tokens) to produce correct
next-step predictions, it MUST put everything needed into the summaries.

See CCT_IMPLEMENTATION_GUIDE.md Section 1.6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SufficiencyProbe(nn.Module):
    """Linear probe that reconstructs step hidden states from summaries.

    Used both as a training loss component (delta) and as the primary
    evaluation metric (R^2).
    """

    def __init__(self, d_summary: int, d_model: int):
        super().__init__()
        self.probe = nn.Linear(d_summary, d_model)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """Reconstruct step hidden state from commitment summary.

        Args:
            summary: (batch, d_summary)
        Returns:
            reconstructed: (batch, d_model)
        """
        return self.probe(summary)


def compute_cct_loss(
    # Standard autoregressive loss
    lm_logits: torch.Tensor,  # (batch, seq_len, vocab_size)
    labels: torch.Tensor,  # (batch, seq_len)
    # Per-step validity: predictions using ONLY summaries as context
    validity_logits: torch.Tensor | None = None,  # (batch, step_len, vocab_size)
    validity_labels: torch.Tensor | None = None,  # (batch, step_len)
    # Conclusion from premises: predict final answer from summaries only
    conclusion_logits: torch.Tensor | None = None,  # (batch, answer_len, vocab_size)
    conclusion_labels: torch.Tensor | None = None,  # (batch, answer_len)
    # Sufficiency probe
    summary: torch.Tensor | None = None,  # (batch, d_summary)
    step_hidden_mean: torch.Tensor | None = None,  # (batch, d_model)
    sufficiency_probe: SufficiencyProbe | None = None,
    # Weights from curriculum
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor | float]:
    """Compute the full CCT loss.

    During Phase 1 (awareness), only L_standard is active.
    As training progresses, the other components ramp up.

    Returns dict with 'total' (Tensor for backprop) and individual
    component values (float, for logging).
    """
    if weights is None:
        weights = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}

    # 1. Standard autoregressive loss (always active)
    L_standard = F.cross_entropy(
        lm_logits.reshape(-1, lm_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    total = weights["alpha"] * L_standard

    # 2. Per-step validity loss
    L_validity = torch.tensor(0.0, device=lm_logits.device)
    if weights["beta"] > 0 and validity_logits is not None and validity_labels is not None:
        L_validity = F.cross_entropy(
            validity_logits.reshape(-1, validity_logits.size(-1)),
            validity_labels.reshape(-1),
            ignore_index=-100,
        )
        total = total + weights["beta"] * L_validity

    # 3. Conclusion from premises loss
    L_conclusion = torch.tensor(0.0, device=lm_logits.device)
    if weights["gamma"] > 0 and conclusion_logits is not None and conclusion_labels is not None:
        L_conclusion = F.cross_entropy(
            conclusion_logits.reshape(-1, conclusion_logits.size(-1)),
            conclusion_labels.reshape(-1),
            ignore_index=-100,
        )
        total = total + weights["gamma"] * L_conclusion

    # 4. Sufficiency probe loss
    L_sufficiency = torch.tensor(0.0, device=lm_logits.device)
    if (
        weights["delta"] > 0
        and summary is not None
        and step_hidden_mean is not None
        and sufficiency_probe is not None
    ):
        reconstructed = sufficiency_probe(summary)
        L_sufficiency = F.mse_loss(reconstructed, step_hidden_mean.detach())
        total = total + weights["delta"] * L_sufficiency

    return {
        "total": total,
        "standard": L_standard.item(),
        "validity": L_validity.item(),
        "conclusion": L_conclusion.item(),
        "sufficiency": L_sufficiency.item(),
    }
