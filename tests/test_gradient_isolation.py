"""
Unit tests for gradient isolation.

Tests:
  1. After detach, gradient of step K params w.r.t. step K+1 loss = 0
  2. Forward pass values are preserved (detach doesn't change values)
  3. Step K's own loss still produces gradients for step K params
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest
from src.model.gradient_isolation import isolate_gradient
from src.model.commitment_head import CommitmentHead


def test_values_preserved():
    """Detach preserves tensor values."""
    x = torch.randn(4, 16)
    y = isolate_gradient(x)
    assert torch.equal(x, y)


def test_no_grad_after_detach():
    """Detached tensor has no grad_fn."""
    x = torch.randn(4, 16, requires_grad=True)
    y = x * 2  # has grad_fn
    z = isolate_gradient(y)
    assert y.grad_fn is not None
    assert z.grad_fn is None
    assert not z.requires_grad


def test_gradient_blocked_across_steps():
    """Gradient from step K+1 loss does NOT flow to step K params.

    Simulates two steps:
      step K: linear_k(x_k) -> commitment_head -> summary -> detach
      step K+1: linear_k1(concat(summary, x_k1)) -> loss

    After backprop from step K+1 loss, linear_k should have NO gradient.
    """
    d_model = 32
    d_summary = 8

    # Step K components
    linear_k = nn.Linear(d_model, d_model)
    head = CommitmentHead(d_model=d_model, d_summary=d_summary)

    # Step K+1 components
    linear_k1 = nn.Linear(d_model + d_summary, d_model)

    # Forward step K
    x_k = torch.randn(2, 5, d_model)
    h_k = linear_k(x_k)  # (2, 5, d_model)
    summary = head(h_k)  # (2, d_summary)
    summary_isolated = isolate_gradient(summary)  # DETACH

    # Forward step K+1
    x_k1 = torch.randn(2, d_model)
    input_k1 = torch.cat([summary_isolated, x_k1], dim=-1)  # (2, d_model + d_summary)
    output_k1 = linear_k1(input_k1)
    loss_k1 = output_k1.sum()

    # Backward from step K+1 loss
    loss_k1.backward()

    # Step K+1 params SHOULD have gradients
    assert linear_k1.weight.grad is not None
    assert linear_k1.weight.grad.abs().sum() > 0

    # Step K params should have NO gradients (isolation worked)
    assert linear_k.weight.grad is None, "Gradient leaked through isolation barrier"
    for name, p in head.named_parameters():
        assert p.grad is None, f"Gradient leaked to commitment head param: {name}"


def test_own_step_gradient_works():
    """Step K's own loss DOES produce gradients for step K params.

    The commitment head should still train from its own step's loss.
    """
    d_model = 32
    d_summary = 8

    head = CommitmentHead(d_model=d_model, d_summary=d_summary)
    x = torch.randn(2, 5, d_model)
    summary = head(x)  # (2, d_summary)

    # Loss from step K itself (not downstream)
    target = torch.randn(2, d_summary)
    loss = ((summary - target) ** 2).sum()
    loss.backward()

    for name, p in head.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
