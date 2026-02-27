"""KV-Cache Prefix Generator for summary delivery.

Generates per-layer Key/Value prefix pairs from commitment summaries,
injected via past_key_values. This uses the model's native KV cache
pathway — the same mechanism used for real cached tokens — so pretrained
attention patterns apply naturally.

Architecture per active layer:
    summary(d_summary) → Linear(d_summary, hidden) → ReLU → Linear(hidden, n_heads * 2 * d_head)
    → reshape → K(n_heads, d_head), V(n_heads, d_head)

RoPE is applied to K vectors at the appropriate boundary positions
(Pythia uses partial RoPE: rotary_ndim=16 out of d_head=64).
"""

import torch
import torch.nn as nn
from transformers import DynamicCache


def _rotate_half(x):
    """Rotates half the hidden dims of the input (matches transformers impl)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class PrefixKVGenerator(nn.Module):
    """Generate per-layer KV prefix pairs from commitment summaries.

    Args:
        d_summary: Commitment summary dimension (e.g., 128)
        n_layers: Total number of transformer layers (e.g., 24)
        n_heads: Number of attention heads (e.g., 16)
        d_head: Dimension per attention head (e.g., 64)
        hidden_dim: MLP hidden dimension (default 512)
        active_layers_min: First active layer (default 12). Layers below
            this get zero K/V (no interference with low-level processing).
        rotary_ndim: Number of dimensions that get RoPE (default 16 for Pythia)
        rope_theta: RoPE base frequency (default 10000.0)
    """

    def __init__(
        self,
        d_summary: int,
        n_layers: int,
        n_heads: int,
        d_head: int,
        hidden_dim: int = 512,
        active_layers_min: int = 12,
        rotary_ndim: int = 16,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.active_layers_min = active_layers_min
        self.rotary_ndim = rotary_ndim

        # Per-layer MLPs for active layers
        self.layer_mlps = nn.ModuleDict()
        out_dim = n_heads * 2 * d_head  # K + V for all heads
        for layer_idx in range(active_layers_min, n_layers):
            self.layer_mlps[str(layer_idx)] = nn.Sequential(
                nn.Linear(d_summary, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        # RoPE inv_freq buffer (matches Pythia's computation)
        # inv_freq = 1 / (theta^(2i/rotary_ndim)) for i=0..rotary_ndim/2-1
        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, rotary_ndim, 2, dtype=torch.float32) / rotary_ndim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        n_active = n_layers - active_layers_min
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  PrefixKVGenerator: {n_params:,} params "
              f"({n_active} active layers [{active_layers_min}-{n_layers-1}], "
              f"hidden={hidden_dim})")

    def _apply_rope_to_k(self, k, positions):
        """Apply RoPE to the first rotary_ndim dimensions of K.

        Args:
            k: (B, n_heads, n_prefix, d_head) — raw K vectors
            positions: (n_prefix,) — position indices for each prefix token

        Returns:
            k with RoPE applied: same shape as input
        """
        # Compute cos/sin for the given positions
        # positions: (n_prefix,) → freqs: (n_prefix, rotary_ndim/2)
        freqs = torch.outer(positions.float(), self.inv_freq)  # (n_prefix, rotary_ndim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (n_prefix, rotary_ndim)
        cos = emb.cos()  # (n_prefix, rotary_ndim)
        sin = emb.sin()  # (n_prefix, rotary_ndim)

        # Reshape for broadcasting: (1, 1, n_prefix, rotary_ndim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split K into rotary and pass-through parts
        k_rot = k[..., :self.rotary_ndim]    # (B, n_heads, n_prefix, rotary_ndim)
        k_pass = k[..., self.rotary_ndim:]   # (B, n_heads, n_prefix, d_head - rotary_ndim)

        # Apply rotation
        k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

        return torch.cat([k_embed, k_pass], dim=-1)

    def forward(self, summaries, boundary_positions, batch_size):
        """Generate past_key_values from committed summaries.

        Args:
            summaries: list of (B, d_summary) tensors — committed summaries
            boundary_positions: list of int — position indices for RoPE
            batch_size: int

        Returns:
            past_key_values: DynamicCache with n_layers (K, V) pairs
                K shape: (B, n_heads, n_prefix, d_head)
                V shape: (B, n_heads, n_prefix, d_head)
        """
        n_prefix = len(summaries)
        device = summaries[0].device
        dtype = summaries[0].dtype

        # Stack summaries: (B, n_prefix, d_summary)
        summary_stack = torch.stack(summaries, dim=1)

        # Position tensor for RoPE
        positions = torch.tensor(
            boundary_positions, dtype=torch.long, device=device
        )

        cache = DynamicCache()

        for layer_idx in range(self.n_layers):
            if layer_idx < self.active_layers_min:
                # Inactive layer: zero K/V
                k = torch.zeros(
                    batch_size, self.n_heads, n_prefix, self.d_head,
                    device=device, dtype=dtype,
                )
                v = torch.zeros_like(k)
            else:
                # Active layer: MLP → K, V
                mlp = self.layer_mlps[str(layer_idx)]
                # (B, n_prefix, d_summary) → (B, n_prefix, n_heads * 2 * d_head)
                kv_raw = mlp(summary_stack.float()).to(dtype)

                # Reshape: (B, n_prefix, n_heads, 2, d_head)
                kv_raw = kv_raw.view(
                    batch_size, n_prefix, self.n_heads, 2, self.d_head
                )

                # Split into K and V: each (B, n_prefix, n_heads, d_head)
                k = kv_raw[:, :, :, 0, :]  # (B, n_prefix, n_heads, d_head)
                v = kv_raw[:, :, :, 1, :]

                # Transpose to (B, n_heads, n_prefix, d_head)
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                # Apply RoPE to K
                k = self._apply_rope_to_k(k, positions)

            # DynamicCache stores K/V per layer via update()
            # We set key_states and value_states directly
            cache.update(k, v, layer_idx)

        return cache

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
