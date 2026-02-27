"""
Gradient Isolation Barrier: Stop-gradient at step boundaries.

Forward: summary passes through normally.
Backward: gradient flow is controlled by mode parameter.

Modes:
  - "full_detach": No gradient flows through (original behavior).
  - "scaled": Gradients are scaled down by a factor (e.g., 0.1).
  - "none": Full gradient flow (end-to-end training).

See CCT_IMPLEMENTATION_GUIDE.md Section 1.4.
"""

import torch


def isolate_gradient(
    summary: torch.Tensor,
    mode: str = "full_detach",
    scale: float = 0.1,
) -> torch.Tensor:
    """Apply gradient isolation to commitment summary.

    Args:
        summary: (batch, d_summary) commitment summary tensor.
        mode: "full_detach" | "scaled" | "none".
        scale: Gradient scaling factor when mode == "scaled".

    Returns:
        Summary tensor with controlled gradient flow.
    """
    if mode == "full_detach":
        return summary.detach()
    elif mode == "scaled":
        # Straight-through with scaled gradient:
        # Forward: returns summary unchanged.
        # Backward: gradient is multiplied by `scale`.
        return summary * scale + summary.detach() * (1.0 - scale)
    else:  # "none"
        return summary
