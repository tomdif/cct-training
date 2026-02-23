"""
Unit tests for commitment head.

Tests:
  1. Output shape is (batch, d_summary)
  2. Output is L2 normalized (norm = 1)
  3. Gradients flow through commitment head during training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.model.commitment_head import CommitmentHead


def test_output_shape():
    """Commitment head produces correct output shape."""
    head = CommitmentHead(d_model=768, d_summary=16, d_bottleneck=32)
    x = torch.randn(4, 10, 768)  # batch=4, step_length=10, d_model=768
    out = head(x)
    assert out.shape == (4, 16)


def test_output_l2_normalized():
    """Output vectors have unit L2 norm."""
    head = CommitmentHead(d_model=64, d_summary=8)
    x = torch.randn(8, 5, 64)
    out = head(x)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_output_l2_normalized_zero_input():
    """L2 normalization handles near-zero inputs without NaN."""
    head = CommitmentHead(d_model=64, d_summary=8)
    x = torch.zeros(2, 5, 64)
    out = head(x)
    assert not torch.any(torch.isnan(out))


def test_gradients_flow():
    """Gradients flow through the commitment head to its parameters."""
    head = CommitmentHead(d_model=64, d_summary=8)
    x = torch.randn(2, 5, 64)
    out = head(x)
    loss = out.sum()
    loss.backward()
    # Check that all parameters have gradients
    for name, param in head.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_default_bottleneck():
    """Default d_bottleneck is 2 * d_summary."""
    head = CommitmentHead(d_model=768, d_summary=16)
    # First linear should be 768 -> 32
    first_linear = head.projection[0]
    assert first_linear.in_features == 768
    assert first_linear.out_features == 32  # 2 * 16


def test_uses_last_token():
    """Head uses the last token position (not mean pooling)."""
    head = CommitmentHead(d_model=64, d_summary=8)
    x = torch.randn(1, 10, 64)
    out1 = head(x)
    # Change early tokens — output should NOT change
    x2 = x.clone()
    x2[:, 0:5, :] = torch.randn(1, 5, 64)
    out2 = head(x2)
    # Change last token — output SHOULD change
    x3 = x.clone()
    x3[:, -1, :] = torch.randn(1, 64)
    out3 = head(x3)
    assert torch.equal(out1, out2), "Output changed when only early tokens changed"
    assert not torch.equal(out1, out3), "Output unchanged when last token changed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
