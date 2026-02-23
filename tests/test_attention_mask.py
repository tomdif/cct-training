"""
Unit tests for CCT attention mask.

Tests:
  1. Mask shape is correct
  2. Within-step attention is causal
  3. Cross-step token attention is blocked
  4. Summary attention is allowed for all later tokens
  5. Fast and slow implementations agree
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.model.cct_attention import (
    build_cct_attention_mask,
    build_cct_attention_mask_fast,
    assign_tokens_to_steps,
)


def test_step_assignment():
    """Tokens are assigned to correct steps."""
    # 10 tokens, boundary at position 4 -> step 0: [0-4], step 1: [5-9]
    steps = assign_tokens_to_steps(10, [4])
    assert steps == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_step_assignment_multiple():
    """Multiple boundaries."""
    steps = assign_tokens_to_steps(9, [2, 5])
    assert steps == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_mask_shape():
    """Mask has correct 4D shape."""
    mask = build_cct_attention_mask(
        seq_len=10,
        boundary_positions=[4],
        num_prior_summaries=2,
        device=torch.device("cpu"),
        batch_size=3,
    )
    assert mask.shape == (3, 1, 12, 12)  # total = 2 + 10 = 12


def test_mask_values_are_valid():
    """Mask only contains 0.0 and -inf."""
    mask = build_cct_attention_mask(
        seq_len=10,
        boundary_positions=[4],
        num_prior_summaries=2,
        device=torch.device("cpu"),
    )
    mask_2d = mask.squeeze()
    for i in range(mask_2d.shape[0]):
        for j in range(mask_2d.shape[1]):
            val = mask_2d[i, j].item()
            assert val == 0.0 or val == float("-inf"), f"Invalid value {val} at ({i},{j})"


def test_summary_causal():
    """Summary tokens attend causally to each other."""
    mask = build_cct_attention_mask(
        seq_len=6,
        boundary_positions=[2],
        num_prior_summaries=3,
        device=torch.device("cpu"),
    )
    m = mask.squeeze()  # (9, 9)
    # Summary 0 attends only to itself
    assert m[0, 0].item() == 0.0
    assert m[0, 1].item() == float("-inf")
    # Summary 1 attends to 0 and 1
    assert m[1, 0].item() == 0.0
    assert m[1, 1].item() == 0.0
    assert m[1, 2].item() == float("-inf")
    # Summary 2 attends to 0, 1, 2
    assert m[2, 0].item() == 0.0
    assert m[2, 1].item() == 0.0
    assert m[2, 2].item() == 0.0


def test_tokens_attend_to_all_summaries():
    """All tokens can attend to all summary tokens."""
    mask = build_cct_attention_mask(
        seq_len=6,
        boundary_positions=[2],
        num_prior_summaries=2,
        device=torch.device("cpu"),
    )
    m = mask.squeeze()  # (8, 8)
    # Every token row (indices 2..7) should have 0.0 in columns 0..1
    for i in range(2, 8):
        assert m[i, 0].item() == 0.0, f"Token {i-2} can't attend to summary 0"
        assert m[i, 1].item() == 0.0, f"Token {i-2} can't attend to summary 1"


def test_within_step_causal():
    """Tokens within the same step have causal (lower-triangular) attention."""
    mask = build_cct_attention_mask(
        seq_len=6,
        boundary_positions=[2],  # step 0: [0,1,2], step 1: [3,4,5]
        num_prior_summaries=0,
        device=torch.device("cpu"),
    )
    m = mask.squeeze()
    # Step 0: tokens 0,1,2
    assert m[0, 0].item() == 0.0  # token 0 -> token 0
    assert m[1, 0].item() == 0.0  # token 1 -> token 0
    assert m[1, 1].item() == 0.0  # token 1 -> token 1
    assert m[2, 0].item() == 0.0  # token 2 -> token 0
    assert m[2, 1].item() == 0.0  # token 2 -> token 1
    assert m[2, 2].item() == 0.0  # token 2 -> token 2
    # Step 1: tokens 3,4,5
    assert m[3, 3].item() == 0.0
    assert m[4, 3].item() == 0.0
    assert m[4, 4].item() == 0.0
    assert m[5, 3].item() == 0.0
    assert m[5, 4].item() == 0.0
    assert m[5, 5].item() == 0.0


def test_cross_step_tokens_blocked():
    """Tokens in step 1 CANNOT attend to tokens in step 0."""
    mask = build_cct_attention_mask(
        seq_len=6,
        boundary_positions=[2],  # step 0: [0,1,2], step 1: [3,4,5]
        num_prior_summaries=0,
        device=torch.device("cpu"),
    )
    m = mask.squeeze()
    # Step 1 tokens (3,4,5) should NOT attend to step 0 tokens (0,1,2)
    for i in [3, 4, 5]:
        for j in [0, 1, 2]:
            assert m[i, j].item() == float("-inf"), (
                f"Token {i} (step 1) can attend to token {j} (step 0)"
            )


def test_no_future_attention():
    """No token attends to future tokens (causal constraint)."""
    mask = build_cct_attention_mask(
        seq_len=6,
        boundary_positions=[2],
        num_prior_summaries=1,
        device=torch.device("cpu"),
    )
    m = mask.squeeze()
    total = m.shape[0]
    for i in range(total):
        for j in range(i + 1, total):
            assert m[i, j].item() == float("-inf"), (
                f"Position {i} attends to future position {j}"
            )


def test_no_boundaries_is_standard_causal():
    """With no step boundaries, mask reduces to standard causal attention."""
    mask = build_cct_attention_mask(
        seq_len=6,
        boundary_positions=[],
        num_prior_summaries=0,
        device=torch.device("cpu"),
    )
    m = mask.squeeze()
    # Should be lower-triangular: m[i,j] = 0 if j<=i, -inf otherwise
    for i in range(6):
        for j in range(6):
            expected = 0.0 if j <= i else float("-inf")
            assert m[i, j].item() == expected


def test_fast_matches_slow():
    """Vectorized implementation matches the loop-based version."""
    kwargs = dict(
        seq_len=15,
        boundary_positions=[4, 9],
        num_prior_summaries=3,
        device=torch.device("cpu"),
        batch_size=1,
    )
    slow = build_cct_attention_mask(**kwargs).squeeze()
    fast = build_cct_attention_mask_fast(**kwargs).squeeze()
    assert torch.equal(slow, fast), "Fast and slow masks differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
