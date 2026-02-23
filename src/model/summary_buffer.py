"""
Summary Buffer: Ordered sequence of commitment summary vectors.

This IS the working memory (F19). At inference time it replaces the KV cache
for cross-step attention. During training it holds the detached summaries
that subsequent steps attend to.

The buffer stores summaries as embeddings that can be prepended to the
transformer's input or injected via the attention mask.

Supports two decoder types:
  - "linear": Single linear projection d_summary -> d_model (original)
  - "mlp": 2-layer MLP with GELU for nonlinear decoding
"""

import torch
import torch.nn as nn


class SummaryBuffer:
    """Stores commitment summaries for cross-step attention.

    During a forward pass over a multi-step sequence, each step produces
    a commitment summary. This buffer collects them so later steps can
    attend to all prior summaries.
    """

    def __init__(
        self,
        d_summary: int,
        d_model: int,
        device: torch.device | None = None,
        decoder_type: str = "linear",
        decoder_bottleneck: int | None = None,
    ):
        """
        Args:
            d_summary: Dimension of commitment summaries.
            d_model: Dimension of transformer hidden states (for up-projection).
            device: Torch device.
            decoder_type: "linear" (original) or "mlp" (nonlinear decoder).
            decoder_bottleneck: MLP hidden dim (default: d_summary * 8).
        """
        self.d_summary = d_summary
        self.d_model = d_model
        self.device = device or torch.device("cpu")

        if decoder_type == "mlp":
            if decoder_bottleneck is None:
                decoder_bottleneck = d_summary * 8
            self.up_project = nn.Sequential(
                nn.Linear(d_summary, decoder_bottleneck),
                nn.GELU(),
                nn.Linear(decoder_bottleneck, d_model),
            ).to(self.device)
        else:
            # Original linear decoder (backward compatible)
            self.up_project = nn.Linear(d_summary, d_model, bias=False).to(self.device)

        # Storage: list of (batch, d_summary) tensors
        self.summaries: list[torch.Tensor] = []

    def clear(self):
        """Reset buffer for a new sequence."""
        self.summaries = []

    def append(self, summary: torch.Tensor):
        """Add a commitment summary to the buffer.

        Args:
            summary: (batch, d_summary) — must already be detached.
        """
        self.summaries.append(summary)

    @property
    def num_summaries(self) -> int:
        return len(self.summaries)

    def get_summary_embeddings(self) -> torch.Tensor | None:
        """Get all stored summaries as transformer-compatible embeddings.

        Returns:
            (batch, num_summaries, d_model) tensor of up-projected summaries,
            or None if buffer is empty.
        """
        if not self.summaries:
            return None

        # Stack: (num_summaries, batch, d_summary) -> (batch, num_summaries, d_summary)
        stacked = torch.stack(self.summaries, dim=1)
        # Up-project to d_model space
        return self.up_project(stacked)  # (batch, num_summaries, d_model)

    def get_raw_summaries(self) -> torch.Tensor | None:
        """Get all stored summaries in raw d_summary space.

        Returns:
            (batch, num_summaries, d_summary) or None if empty.
        """
        if not self.summaries:
            return None
        return torch.stack(self.summaries, dim=1)
