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
    """Projects step-final hidden states to d_summary commitment vectors.

    With n_summary_tokens > 1, produces K separate summary vectors per step.
    Each gets independently up-projected into a KV entry, giving the attention
    mechanism K different value vectors to choose from.
    """

    def __init__(
        self,
        d_model: int,
        d_summary: int,
        d_bottleneck: int | None = None,
        use_tanh: bool = True,
        use_l2_norm: bool = True,
        noise_injection: bool = False,
        n_summary_tokens: int = 1,
        recurrent: bool = False,
    ):
        """
        Args:
            d_model: Hidden dimension of the transformer.
            d_summary: Output commitment summary dimension (per token).
            d_bottleneck: Intermediate bottleneck dimension (default: 2 * d_summary).
            use_tanh: Use Tanh activation (default True for backward compat).
            use_l2_norm: L2-normalize output to unit sphere (default True).
            noise_injection: Add calibrated Gaussian noise to output.
            n_summary_tokens: Number of summary tokens to produce per step (K).
            recurrent: If True, accept prev_summary to chain information forward.
        """
        super().__init__()
        self.n_summary_tokens = n_summary_tokens
        self.d_summary = d_summary
        self.recurrent = recurrent

        if d_bottleneck is None:
            d_bottleneck = d_summary * 2
        self.d_bottleneck = d_bottleneck

        layers = [
            nn.Linear(d_model, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
        ]
        if use_tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.GELU())
        layers.append(nn.Linear(d_bottleneck, d_summary * n_summary_tokens))

        self.projection = nn.Sequential(*layers)

        if recurrent:
            # GRU cell replaces the simple additive gate.
            # Input: bottleneck activation projected to d_summary
            # Hidden state: prev_summary (d_summary, L2-normalized)
            self.gru_input_proj = nn.Linear(d_bottleneck, d_summary)
            self.gru = nn.GRUCell(input_size=d_summary, hidden_size=d_summary)
            # Output projection: GRU hidden (d_summary) → d_summary * K
            self.gru_output = nn.Linear(d_summary, d_summary * n_summary_tokens)
            # Init update gate bias positive → sigmoid ≈ 0.88 → keep prev state early
            with torch.no_grad():
                self.gru.bias_hh.data[d_summary:2*d_summary].fill_(2.0)

        self.use_l2_norm = use_l2_norm
        self.noise_injection = noise_injection
        self.noise_sigma = 0.0  # Set externally by curriculum

    def forward(
        self,
        step_hidden_states: torch.Tensor,
        prev_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute commitment summary from step hidden states.

        Args:
            step_hidden_states: (batch, step_length, d_model)
                Hidden states from the last transformer layer for one step.
            prev_summary: (batch, d_summary) optional previous step's summary.
                Only used when recurrent=True. Enables chaining information
                across steps: summary_k = f(hidden_k, summary_{k-1}).

        Returns:
            If n_summary_tokens == 1:
                summary: (batch, d_summary)
            If n_summary_tokens > 1:
                summary: (batch, K, d_summary)
        """
        # Use last token's hidden state (like [CLS] pooling at end of step)
        last_token = step_hidden_states[:, -1, :]  # (batch, d_model)

        if self.recurrent:
            # Pre-activation: Linear + LayerNorm + activation → (B, d_bottleneck)
            h = last_token
            for layer in list(self.projection.children())[:-1]:
                h = layer(h)
            # Project to GRU input dimension
            gru_input = self.gru_input_proj(h)  # (B, d_summary)
            self._last_gru_input = gru_input  # saved for multi-hop loss chain rebuild
            # GRU step: prev_summary is the hidden state (L2-normalized)
            if prev_summary is not None:
                gru_h = self.gru(gru_input, prev_summary)  # (B, d_summary)
            else:
                gru_h = self.gru(gru_input)  # zero hidden state
            # Log state change magnitude for monitoring
            if prev_summary is not None:
                self._last_gate_mean = (gru_h - prev_summary).norm(dim=-1).mean().item()
            else:
                self._last_gate_mean = 0.0
            # Output projection: (B, d_summary) → (B, d_summary * K)
            raw = self.gru_output(gru_h)
        else:
            raw = self.projection(last_token)  # (B, d_summary * K)

        if self.n_summary_tokens > 1:
            summary = raw.view(-1, self.n_summary_tokens, self.d_summary)
        else:
            summary = raw

        if self.use_l2_norm:
            summary = F.normalize(summary, dim=-1)  # L2 normalize to unit sphere

        if self.noise_injection and self.training and self.noise_sigma > 0:
            summary = summary + self.noise_sigma * torch.randn_like(summary)

        return summary
