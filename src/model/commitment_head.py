"""
Commitment Head: Bottleneck projection from d_model to d_summary.

Default architecture: Linear -> LayerNorm -> Tanh -> Linear -> L2 Normalize
Noise-injection mode: Linear -> LayerNorm -> GELU -> Linear -> Noise

L2 normalization constrains summaries to a hypersphere (geometric constraint).
Noise injection provides a softer information-theoretic constraint: anything
encoded at scale below σ is destroyed, so the model learns to encode
high-value information at large scale (rate-distortion optimal).

See CCT_IMPLEMENTATION_GUIDE.md Section 1.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommitmentHead(nn.Module):
    """Projects step-final hidden states to d_summary commitment vectors."""

    def __init__(
        self,
        d_model: int,
        d_summary: int,
        d_bottleneck: int | None = None,
        use_tanh: bool = True,
        use_l2_norm: bool = True,
        noise_injection: bool = False,
    ):
        """
        Args:
            d_model: Hidden dimension of the transformer.
            d_summary: Output commitment summary dimension.
            d_bottleneck: Intermediate bottleneck dimension (default: 2 * d_summary).
            use_tanh: Use Tanh activation (default True for backward compat).
            use_l2_norm: L2-normalize output to unit sphere (default True).
            noise_injection: Add calibrated Gaussian noise to output.
        """
        super().__init__()
        if d_bottleneck is None:
            d_bottleneck = d_summary * 2

        layers = [
            nn.Linear(d_model, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
        ]
        if use_tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.GELU())
        layers.append(nn.Linear(d_bottleneck, d_summary))

        self.projection = nn.Sequential(*layers)
        self.d_summary = d_summary
        self.use_l2_norm = use_l2_norm
        self.noise_injection = noise_injection
        self.noise_sigma = 0.0  # Set externally by curriculum

    def forward(self, step_hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute commitment summary from step hidden states.

        Args:
            step_hidden_states: (batch, step_length, d_model)
                Hidden states from the last transformer layer for one step.

        Returns:
            summary: (batch, d_summary) — commitment summary.
        """
        # Use last token's hidden state (like [CLS] pooling at end of step)
        last_token = step_hidden_states[:, -1, :]  # (batch, d_model)
        summary = self.projection(last_token)  # (batch, d_summary)

        if self.use_l2_norm:
            summary = F.normalize(summary, dim=-1)  # L2 normalize to unit sphere

        if self.noise_injection and self.training and self.noise_sigma > 0:
            summary = summary + self.noise_sigma * torch.randn_like(summary)

        return summary
