"""
Pseudo-token decoder: expands a single d_summary vector into multiple
d_model-dimensional pseudo-token embeddings for attention-based delivery.

Unlike K>1 summary tokens (which are independent pooling operations from the
commitment head), this is a learned expansion: the commitment head compresses
into d_summary, then this decoder reconstructs diverse, attention-compatible
representations. Different attention heads and query positions can attend to
different pseudo-tokens, solving the "one vector for all queries" bottleneck.

Architecture: 2-layer MLP with GELU activation.
  d_summary → hidden → n_pseudo_tokens × d_model → reshape (B, n_pseudo, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PseudoTokenDecoder(nn.Module):
    def __init__(self, d_summary, d_model, n_pseudo_tokens=8, hidden_dim=512, device=None):
        super().__init__()
        self.d_summary = d_summary
        self.d_model = d_model
        self.n_pseudo_tokens = n_pseudo_tokens

        self.fc1 = nn.Linear(d_summary, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, n_pseudo_tokens * d_model, bias=True)

        # Xavier init scaled down to prevent dominating early training
        nn.init.xavier_normal_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

        if device is not None:
            self.to(device)

    def forward(self, summary):
        """Expand summary into pseudo-token embeddings.

        Args:
            summary: (B, d_summary) compressed summary vector
        Returns:
            (B, n_pseudo_tokens, d_model) pseudo-token embeddings
        """
        h = F.gelu(self.fc1(summary.float()))  # (B, hidden)
        out = self.fc2(h)                       # (B, n_pseudo * d_model)
        return out.view(summary.shape[0], self.n_pseudo_tokens, self.d_model)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
