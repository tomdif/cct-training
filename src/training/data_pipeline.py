"""
Step Boundary Annotation

Inserts STEP tokens at step boundaries in training data.
- Fixed mode: every N tokens
- Semantic mode: at sentence/paragraph boundaries (Tier 2+)

See CCT_IMPLEMENTATION_GUIDE.md Section 1.1.
"""

from transformers import AutoTokenizer

STEP_TOKEN = "<STEP>"


def extend_tokenizer(tokenizer: AutoTokenizer) -> int:
    """Add STEP token to tokenizer. Returns the new token's ID."""
    tokenizer.add_special_tokens({"additional_special_tokens": [STEP_TOKEN]})
    step_token_id = tokenizer.convert_tokens_to_ids(STEP_TOKEN)
    return step_token_id


def annotate_step_boundaries(
    token_ids: list[int],
    step_token_id: int,
    mode: str = "fixed",
    step_length: int = 50,
) -> list[int]:
    """Insert step_token_id at step boundaries.

    Args:
        token_ids: Original token sequence.
        step_token_id: ID of the <STEP> token.
        mode: "fixed" (every N tokens) or "semantic" (sentence/paragraph).
        step_length: For fixed mode, tokens per step.

    Returns:
        Token sequence with step_token_id inserted at boundaries.
    """
    if mode == "fixed":
        return _annotate_fixed(token_ids, step_token_id, step_length)
    elif mode == "semantic":
        return _annotate_semantic(token_ids, step_token_id, step_length)
    else:
        raise ValueError(f"Unknown boundary mode: {mode}")


def _annotate_fixed(
    token_ids: list[int],
    step_token_id: int,
    step_length: int,
) -> list[int]:
    """Insert STEP token every step_length tokens."""
    result = []
    for i, tid in enumerate(token_ids):
        result.append(tid)
        # Insert boundary after every step_length tokens (but not at the very end)
        if (i + 1) % step_length == 0 and (i + 1) < len(token_ids):
            result.append(step_token_id)
    return result


def _annotate_semantic(
    token_ids: list[int],
    step_token_id: int,
    step_length: int,
) -> list[int]:
    """Insert STEP token at semantic boundaries (sentence/paragraph).

    Falls back to fixed boundaries if no semantic boundary is found
    within 2x step_length tokens.

    For Tier 2+: detect period/newline tokens as boundary candidates.
    """
    # Placeholder for Tier 2 — Tier 1 uses fixed mode only.
    return _annotate_fixed(token_ids, step_token_id, step_length)


def get_step_boundary_positions(token_ids: list[int], step_token_id: int) -> list[int]:
    """Return indices where STEP tokens appear in the sequence."""
    return [i for i, tid in enumerate(token_ids) if tid == step_token_id]


def get_step_ranges(
    seq_len: int,
    boundary_positions: list[int],
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each step.

    Each step spans from after the previous STEP token to the current one.
    The STEP token itself is at position boundary_positions[i] and belongs
    to the step it terminates.
    """
    ranges = []
    prev = 0
    for bp in boundary_positions:
        ranges.append((prev, bp))  # tokens before the STEP token
        prev = bp + 1  # skip the STEP token itself
    # Last step: remaining tokens after final STEP token
    if prev < seq_len:
        ranges.append((prev, seq_len))
    return ranges
