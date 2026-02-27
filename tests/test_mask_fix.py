"""
Test that the CCT attention mask correctly allows summary visibility.

The critical fix: STEP tokens (which carry injected summaries) must be visible
to the next step's tokens. Before the fix, STEP tokens were assigned to the
step they terminated, making them invisible to the step that needed them.

After the fix, STEP tokens belong to the step they introduce, so causal
attention naturally lets next-step tokens attend to the summary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.cct_attention import (
    assign_tokens_to_steps,
    build_cct_attention_mask,
    build_cct_attention_mask_fast,
)


def test_step_assignment():
    """STEP tokens should belong to the step they introduce (next step)."""
    # seq_len=15, STEP tokens at positions 5 and 11
    steps = assign_tokens_to_steps(15, [5, 11])

    # Content tokens before first STEP: step 0
    assert steps[0] == 0
    assert steps[4] == 0

    # STEP at position 5: should be step 1 (introduces step 1)
    assert steps[5] == 1, f"STEP at pos 5 should be step 1, got {steps[5]}"

    # Content tokens after first STEP: step 1
    assert steps[6] == 1
    assert steps[10] == 1

    # STEP at position 11: should be step 2
    assert steps[11] == 2, f"STEP at pos 11 should be step 2, got {steps[11]}"

    # Content tokens after second STEP: step 2
    assert steps[12] == 2
    assert steps[14] == 2

    print("  PASS: step_assignment")


def test_summary_visible_to_next_step():
    """Next-step tokens must be able to attend to the STEP/summary position."""
    mask_4d = build_cct_attention_mask_fast(
        seq_len=15, boundary_positions=[5, 11],
        num_prior_summaries=0, device=torch.device("cpu"), batch_size=1,
    )
    mask = mask_4d[0, 0]  # (15, 15)

    # Token 6 (step 1) attending to position 5 (summary from step 0): MUST ATTEND
    assert mask[6, 5].item() == 0.0, \
        f"Token 6 should attend to summary at pos 5, got {mask[6, 5].item()}"

    # Token 7 attending to position 5: MUST ATTEND
    assert mask[7, 5].item() == 0.0, \
        f"Token 7 should attend to summary at pos 5, got {mask[7, 5].item()}"

    # Token 10 attending to position 5: MUST ATTEND
    assert mask[10, 5].item() == 0.0, \
        f"Token 10 should attend to summary at pos 5, got {mask[10, 5].item()}"

    # Token 12 (step 2) attending to position 11 (summary from step 1): MUST ATTEND
    assert mask[12, 11].item() == 0.0, \
        f"Token 12 should attend to summary at pos 11, got {mask[12, 11].item()}"

    print("  PASS: summary_visible_to_next_step")


def test_summary_blocked_from_previous_step():
    """Previous-step tokens must NOT see the STEP/summary at the boundary."""
    mask_4d = build_cct_attention_mask_fast(
        seq_len=15, boundary_positions=[5, 11],
        num_prior_summaries=0, device=torch.device("cpu"), batch_size=1,
    )
    mask = mask_4d[0, 0]

    # Token 4 (step 0) attending to position 5 (step 1): MUST BLOCK
    # (also blocked by causality: j=5 > i=4)
    assert mask[4, 5].item() == float("-inf"), \
        f"Token 4 should NOT attend to pos 5, got {mask[4, 5].item()}"

    # Token 3 attending to position 5: MUST BLOCK
    assert mask[3, 5].item() == float("-inf"), \
        f"Token 3 should NOT attend to pos 5, got {mask[3, 5].item()}"

    print("  PASS: summary_blocked_from_previous_step")


def test_cross_step_content_blocked():
    """Content tokens from different steps must NOT see each other."""
    mask_4d = build_cct_attention_mask_fast(
        seq_len=15, boundary_positions=[5, 11],
        num_prior_summaries=0, device=torch.device("cpu"), batch_size=1,
    )
    mask = mask_4d[0, 0]

    # Token 6 (step 1) attending to token 4 (step 0 content): MUST BLOCK
    assert mask[6, 4].item() == float("-inf"), \
        f"Token 6 should NOT attend to token 4, got {mask[6, 4].item()}"

    # Token 6 (step 1) attending to token 0 (step 0 content): MUST BLOCK
    assert mask[6, 0].item() == float("-inf"), \
        f"Token 6 should NOT attend to token 0, got {mask[6, 0].item()}"

    # Token 12 (step 2) attending to token 10 (step 1 content): MUST BLOCK
    assert mask[12, 10].item() == float("-inf"), \
        f"Token 12 should NOT attend to token 10, got {mask[12, 10].item()}"

    # Token 12 (step 2) attending to token 5 (step 1 summary): MUST BLOCK (Markov)
    assert mask[12, 5].item() == float("-inf"), \
        f"Token 12 should NOT attend to step 0 summary at pos 5, got {mask[12, 5].item()}"

    print("  PASS: cross_step_content_blocked")


def test_within_step_causal():
    """Within-step causal attention must work normally."""
    mask_4d = build_cct_attention_mask_fast(
        seq_len=15, boundary_positions=[5, 11],
        num_prior_summaries=0, device=torch.device("cpu"), batch_size=1,
    )
    mask = mask_4d[0, 0]

    # Step 0: token 3 attends to tokens 0, 1, 2, 3 (causal)
    for j in range(4):
        assert mask[3, j].item() == 0.0, \
            f"Token 3 should attend to token {j}, got {mask[3, j].item()}"

    # Step 1: token 8 attends to tokens 5, 6, 7, 8 (summary + content, causal)
    for j in [5, 6, 7, 8]:
        assert mask[8, j].item() == 0.0, \
            f"Token 8 should attend to token {j}, got {mask[8, j].item()}"

    # Step 1: token 8 should NOT attend to token 9 (future, not causal)
    assert mask[8, 9].item() == float("-inf"), \
        f"Token 8 should NOT attend to future token 9, got {mask[8, 9].item()}"

    print("  PASS: within_step_causal")


def test_slow_and_fast_match():
    """Slow (loop) and fast (vectorized) mask builders must produce identical masks."""
    for boundary_positions in [[5, 11], [3, 7, 11], [10], []]:
        mask_slow = build_cct_attention_mask(
            seq_len=15, boundary_positions=boundary_positions,
            num_prior_summaries=0, device=torch.device("cpu"), batch_size=1,
        )
        mask_fast = build_cct_attention_mask_fast(
            seq_len=15, boundary_positions=boundary_positions,
            num_prior_summaries=0, device=torch.device("cpu"), batch_size=1,
        )
        # Compare: both -inf and 0.0 should match
        slow_attend = (mask_slow[0, 0] == 0.0)
        fast_attend = (mask_fast[0, 0] == 0.0)
        assert torch.equal(slow_attend, fast_attend), \
            f"Slow and fast masks differ for boundaries={boundary_positions}"

    print("  PASS: slow_and_fast_match")


def test_step_sees_exactly_one_summary():
    """Each step should see exactly the summary at the start of its step (Markov)."""
    mask_4d = build_cct_attention_mask_fast(
        seq_len=18, boundary_positions=[5, 11],
        num_prior_summaries=0, device=torch.device("cpu"), batch_size=1,
    )
    mask = mask_4d[0, 0]

    # Step 1 (positions 5-10) can see summary at pos 5 but NOT pos 11
    for pos in [6, 7, 8, 9, 10]:
        assert mask[pos, 5].item() == 0.0, \
            f"Step 1 token {pos} should see summary at 5"
        assert mask[pos, 11].item() == float("-inf"), \
            f"Step 1 token {pos} should NOT see summary at 11"

    # Step 2 (positions 11-17) can see summary at pos 11 but NOT pos 5
    for pos in [12, 13, 14, 15, 16, 17]:
        assert mask[pos, 11].item() == 0.0, \
            f"Step 2 token {pos} should see summary at 11"
        assert mask[pos, 5].item() == float("-inf"), \
            f"Step 2 token {pos} should NOT see summary at 5"

    print("  PASS: step_sees_exactly_one_summary (Markov property)")


if __name__ == "__main__":
    print("Testing CCT attention mask fix...")
    test_step_assignment()
    test_summary_visible_to_next_step()
    test_summary_blocked_from_previous_step()
    test_cross_step_content_blocked()
    test_within_step_causal()
    test_slow_and_fast_match()
    test_step_sees_exactly_one_summary()
    print("\nAll tests PASSED.")
