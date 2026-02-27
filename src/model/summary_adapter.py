"""
Gated cross-attention adapter for frozen-model CCT.

Each transformer layer gets a small cross-attention module that reads from
a bank of past committed summaries. Gates are initialized near zero so the
model starts at baseline quality. Only adapter parameters are trained;
the base model stays frozen.

Design:
  - Query: current hidden state (B, S, d_model) -> (B, S, d_adapter)
  - Key/Value: summary bank (B, N, d_summary) -> (B, N, d_adapter)
  - Cross-attention: softmax(Q @ K^T / sqrt(d_adapter)) @ V
  - Output: project back to d_model, multiply by sigmoid(gate), add to residual
  - Gate init: -5 -> sigmoid(-5) ~ 0.007, so adapter starts near-identity
"""

import math
import torch
import torch.nn as nn

from src.model.model_utils import get_model_layers


class SummaryAdapter(nn.Module):
    def __init__(self, d_model, d_summary, n_layers, d_adapter=64, n_heads=1, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_summary = d_summary
        self.n_layers = n_layers
        self.d_adapter = d_adapter
        self.n_heads = n_heads
        assert d_adapter % n_heads == 0, "d_adapter must be divisible by n_heads"
        self.d_head = d_adapter // n_heads

        # Per-layer cross-attention projections
        self.q_projs = nn.ModuleList([
            nn.Linear(d_model, d_adapter, bias=False) for _ in range(n_layers)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(d_summary, d_adapter, bias=False) for _ in range(n_layers)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(d_summary, d_adapter, bias=False) for _ in range(n_layers)
        ])
        self.o_projs = nn.ModuleList([
            nn.Linear(d_adapter, d_model, bias=False) for _ in range(n_layers)
        ])

        # Per-layer gates, initialized to -2 -> sigmoid(-2) ~ 0.12
        # -5.0 was too conservative: sigmoid'(-5) ~ 0.007 starves gradient flow
        # -2.0 gives ~15x more gradient while o_proj=0 keeps output near-identity
        self.gates = nn.Parameter(torch.full((n_layers,), -2.0))

        # Initialize all projections with small random weights (Xavier-scale)
        # Note: o_proj CANNOT be zero-init — it blocks all gradient to Q/K/V/gates
        # (gate * 0 = 0 → zero gate gradient; o_proj^T = 0 → zero Q/K/V gradient)
        # Small Xavier + small gate (sigmoid(-2)=0.12) keeps output near-identity
        for proj_list in [self.q_projs, self.k_projs, self.v_projs, self.o_projs]:
            for proj in proj_list:
                nn.init.xavier_normal_(proj.weight, gain=0.1)

        # Runtime state: summary bank set before each forward pass
        self._summary_bank = None  # (B, N, d_summary) or None
        self._hooks = []

        if device is not None:
            self.to(device)

    def set_summary_bank(self, summary_bank):
        """Set the summary bank for the next forward pass.

        Args:
            summary_bank: (B, N_summaries, d_summary) tensor of all past
                committed summaries, or None for no summaries.
        """
        self._summary_bank = summary_bank

    def clear(self):
        """Clear the summary bank after forward pass."""
        self._summary_bank = None

    def register_hooks(self, model):
        """Register forward hooks on each transformer layer."""
        self.remove_hooks()
        for i in range(self.n_layers):
            hook = get_model_layers(model)[i].register_forward_hook(
                self._make_hook(i)
            )
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_hook(self, layer_idx):
        """Create a forward hook that adds gated cross-attention output to residual."""
        def hook(module, args, output):
            if self._summary_bank is None:
                return output

            # output is a tuple: (hidden_states, ...) for GPTNeoX layers
            if isinstance(output, tuple):
                hidden = output[0]  # (B, S, d_model)
            else:
                hidden = output

            B, S, D = hidden.shape
            bank = self._summary_bank  # (B, N, d_summary)

            if bank.device != hidden.device:
                bank = bank.to(hidden.device)

            # Cast to float32 for attention computation
            hidden_f32 = hidden.float()
            bank_f32 = bank.float()

            # Cross-attention
            Q = self.q_projs[layer_idx](hidden_f32)   # (B, S, d_adapter)
            K = self.k_projs[layer_idx](bank_f32)      # (B, N, d_adapter)
            V = self.v_projs[layer_idx](bank_f32)      # (B, N, d_adapter)

            # Multi-head reshape: (B, S, n_heads, d_head) -> (B, n_heads, S, d_head)
            Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
            N = K.shape[1]
            K = K.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            V = V.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

            # Scaled dot-product attention
            scale = math.sqrt(self.d_head)
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, S, N)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_weights, V)  # (B, H, S, d_head)

            # Reshape back: (B, H, S, d_head) -> (B, S, d_adapter)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_adapter)

            # Output projection + gating
            residual = self.o_projs[layer_idx](attn_out)  # (B, S, d_model)
            gate = torch.sigmoid(self.gates[layer_idx])
            residual = gate * residual

            # Add to hidden states, cast back to original dtype
            new_hidden = hidden + residual.to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        return hook

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
