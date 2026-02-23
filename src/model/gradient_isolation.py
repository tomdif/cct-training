"""
Gradient Isolation Barrier: Stop-gradient at step boundaries.

Forward: summary passes through normally.
Backward: no gradient flows through summary to prior step parameters.

Implementation: summary.detach()

See CCT_IMPLEMENTATION_GUIDE.md Section 1.4.
"""

import torch

def isolate_gradient(summary: torch.Tensor) -> torch.Tensor:
    """Detach summary from computation graph for gradient isolation."""
    return summary.detach()
