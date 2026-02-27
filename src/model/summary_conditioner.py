"""
Summary Conditioner: FiLM + Gated Residual Injection.

Delivers summary information to the transformer by modulating LayerNorm
outputs (FiLM) and injecting into the residual stream at every layer.
Bypasses attention entirely — summaries don't compete in softmax.

Design:
  For each layer l:
    FiLM:      hidden = gamma_l * LayerNorm(x) + beta_l
    Residual:  hidden = hidden + sigmoid(gate_l) * project_l(summary)

  where gamma_l, beta_l = Linear_l(summary)

Initialized to identity (gamma=1, beta=0, gate~0) so the model starts
as the unchanged pretrained baseline. Training learns to deviate.
"""

import torch
import torch.nn as nn

from src.model.model_utils import get_model_layers


class SummaryConditioner(nn.Module):
    """FiLM conditioning + gated residual injection for summary delivery.

    For each transformer layer:
      - FiLM: modulates input_layernorm output before attention
        gamma, beta = project(summary); out = gamma * LN(x) + beta
      - Residual: adds gated projection of summary to layer output
        out = out + sigmoid(gate) * project(summary)

    ~9.5M learnable parameters for 24-layer, d_model=1024, d_summary=128.
    """

    def __init__(self, d_summary, d_model, n_layers, device=None):
        super().__init__()
        self.d_summary = d_summary
        self.d_model = d_model
        self.n_layers = n_layers

        # FiLM per layer: summary -> (gamma, beta) for LayerNorm modulation
        self.film_projections = nn.ModuleList([
            nn.Linear(d_summary, 2 * d_model)
            for _ in range(n_layers)
        ])
        # Initialize to identity: gamma=1, beta=0
        for proj in self.film_projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            proj.bias.data[:d_model] = 1.0  # gamma init = 1

        # Residual injection per layer: summary -> d_model vector
        self.residual_projections = nn.ModuleList([
            nn.Linear(d_summary, d_model)
            for _ in range(n_layers)
        ])
        # Initialize to zero (no injection initially)
        for proj in self.residual_projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

        # Per-layer gates: sigmoid(-5) ~ 0.007, nearly off at start
        self.residual_gates = nn.Parameter(torch.full((n_layers,), -5.0))

        # Runtime state
        self._current_summary = None  # (B, d_summary) or None
        self._hooks = []

        if device is not None:
            self.to(device)

    def set_summary(self, summary):
        """Set conditioning summary for the next forward pass.

        Args:
            summary: (B, d_summary) aggregated summary vector, or None.
        """
        self._current_summary = summary

    def clear(self):
        """Disable conditioning (no summary influence)."""
        self._current_summary = None

    def register_hooks(self, model):
        """Register forward hooks on GPTNeoX layers.

        - FiLM hook on input_layernorm (modulates attention input)
        - Residual hook on the full layer (adds to residual stream)
        """
        for i in range(self.n_layers):
            h_film = get_model_layers(model)[i].input_layernorm.register_forward_hook(
                self._make_film_hook(i)
            )
            h_res = get_model_layers(model)[i].register_forward_hook(
                self._make_residual_hook(i)
            )
            self._hooks.extend([h_film, h_res])

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_film_hook(self, layer_idx):
        """FiLM hook: modulates input_layernorm output with gamma * x + beta."""
        def hook(module, input, output):
            s = self._current_summary
            if s is None:
                return output
            # output: (B, S, D) — layernorm output going into attention
            film = self.film_projections[layer_idx](s)  # (B, 2*D)
            gamma = film[:, :self.d_model].unsqueeze(1)  # (B, 1, D)
            beta = film[:, self.d_model:].unsqueeze(1)   # (B, 1, D)
            return gamma.to(output.dtype) * output + beta.to(output.dtype)
        return hook

    def _make_residual_hook(self, layer_idx):
        """Residual hook: adds gated summary projection to layer output."""
        def hook(module, input, output):
            s = self._current_summary
            if s is None:
                return output
            proj = self.residual_projections[layer_idx](s)  # (B, D)
            gate = torch.sigmoid(self.residual_gates[layer_idx])
            inj = (gate * proj).unsqueeze(1)  # (B, 1, D)
            # GPTNeoX layer returns tuple: (hidden_states, ...)
            if isinstance(output, tuple):
                return (output[0] + inj.to(output[0].dtype),) + output[1:]
            return output + inj.to(output.dtype)
        return hook
