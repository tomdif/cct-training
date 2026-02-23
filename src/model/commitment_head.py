"""
Commitment Head: Bottleneck projection from d_model to d_summary.

Architecture: Linear -> LayerNorm -> Tanh -> Linear -> L2 Normalize

The bottleneck forces the model to compress all cross-step-relevant
information into a fixed-size vector. L2 normalization ensures summaries
live on a hypersphere, stabilizing attention and making sufficiency
measurement well-defined.

See CCT_IMPLEMENTATION_GUIDE.md Section 1.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommitmentHead(nn.Module):
    """Projects step-final hidden states to d_summary commitment vectors."""

    def __init__(self, d_model: int, d_summary: int, d_bottleneck: int | None = None):
        """
        Args:
            d_model: Hidden dimension of the transformer.
            d_summary: Output commitment summary dimension.
            d_bottleneck: Intermediate bottleneck dimension (default: 2 * d_summary).
        """
        super().__init__()
        if d_bottleneck is None:
            d_bottleneck = d_summary * 2

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
            nn.Tanh(),
            nn.Linear(d_bottleneck, d_summary),
        )
        self.d_summary = d_summary

    def forward(self, step_hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute commitment summary from step hidden states.

        Args:
            step_hidden_states: (batch, step_length, d_model)
                Hidden states from the last transformer layer for one step.

        Returns:
            summary: (batch, d_summary) — L2-normalized commitment summary.
        """
        # Use last token's hidden state (like [CLS] pooling at end of step)
        last_token = step_hidden_states[:, -1, :]  # (batch, d_model)
        summary = self.projection(last_token)  # (batch, d_summary)
        summary = F.normalize(summary, dim=-1)  # L2 normalize to unit sphere
        return summary
