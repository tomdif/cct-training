"""
Summary as logit bias — bypasses attention entirely.

Summary bank → mean pool → MLP → vocabulary-sized logit adjustment.
The summary doesn't inject content into hidden states — it directly
shifts the output distribution. Near-zero at init via learned magnitude.

If this shows improvement, summaries carry useful information.
If not, the commitment head isn't learning useful representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SummaryLogitBias(nn.Module):
    def __init__(self, d_summary, vocab_size, hidden_dim=256, device=None):
        super().__init__()
        self.d_summary = d_summary
        self.vocab_size = vocab_size

        self.fc1 = nn.Linear(d_summary, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Small Xavier init
        nn.init.xavier_normal_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight, gain=0.1)

        # Learnable magnitude: exp(-4) ≈ 0.018, near-zero at start
        self.log_magnitude = nn.Parameter(torch.tensor(-4.0))

        if device is not None:
            self.to(device)

    def forward(self, bank):
        """Compute logit bias from summary bank.

        Args:
            bank: (B, N, d_summary) summary bank, or None
        Returns:
            (B, vocab_size) logit bias, broadcastable to (B, S, V) via unsqueeze(1)
        """
        if bank is None or bank.shape[1] == 0:
            B = bank.shape[0] if bank is not None else 1
            return torch.zeros(B, self.vocab_size, device=self.fc1.weight.device)

        pooled = bank.float().mean(dim=1)  # (B, d_summary)
        h = F.gelu(self.fc1(pooled))       # (B, hidden_dim)
        raw = self.fc2(h)                   # (B, vocab_size)
        bias = torch.exp(self.log_magnitude) * torch.tanh(raw)  # (B, vocab_size)
        return bias

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
