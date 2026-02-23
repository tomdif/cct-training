"""
CCT Attention Mask Construction

Builds the modified attention mask that:
- Allows causal within-step attention (standard)
- Blocks cross-step token-level attention
- Allows cross-step summary-level attention

For Pythia (GPTNeoX): we pass a 4D float mask to the model loaded with
attn_implementation="eager". The mask uses 0.0 for "attend" and -inf for
"mask out", matching HuggingFace's convention where the mask is ADDED to
attention logits before softmax.

See CCT_IMPLEMENTATION_GUIDE.md Section 1.2.
"""

import torch


def assign_tokens_to_steps(
    seq_len: int,
    boundary_positions: list[int],
) -> list[int]:
    """Assign each token position to its step index.

    Args:
        seq_len: Total sequence length (excluding prepended summaries).
        boundary_positions: Indices of STEP tokens in the sequence.

    Returns:
        List of length seq_len where element i is the step index for token i.
        STEP tokens belong to the step they terminate.
    """
    boundary_set = set(boundary_positions)
    step_of = []
    current_step = 0
    for pos in range(seq_len):
        step_of.append(current_step)
        if pos in boundary_set:
            current_step += 1
    return step_of


def build_cct_attention_mask(
    seq_len: int,
    boundary_positions: list[int],
    num_prior_summaries: int,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Build the 4D CCT attention mask for HuggingFace eager attention.

    The input to the transformer is structured as:
        [summary_1, ..., summary_K, token_1, token_2, ..., token_L]

    Where summary_* are up-projected commitment summaries from prior steps
    prepended as virtual tokens, and token_* are the actual input tokens
    (including STEP boundary tokens).

    Attention rules:
        1. Summary tokens attend causally to prior summaries only.
        2. Tokens within the same step attend causally to each other.
        3. ALL tokens can attend to ALL prior summary tokens.
        4. Tokens CANNOT attend to tokens from earlier steps (only summaries).

    Args:
        seq_len: Number of actual tokens (excluding prepended summaries).
        boundary_positions: Indices of STEP tokens within the seq_len tokens.
        num_prior_summaries: Number of summary tokens prepended.
        device: Torch device.
        batch_size: Batch size (mask is broadcast over heads).

    Returns:
        4D mask of shape (batch_size, 1, total_len, total_len).
        Values: 0.0 = attend, -inf = mask out.
        The head dimension is 1 (broadcast to all heads).
    """
    total_len = num_prior_summaries + seq_len
    # Start with everything masked (-inf)
    mask = torch.full(
        (total_len, total_len),
        float("-inf"),
        device=device,
    )

    # --- Region 1: Summary-to-summary (top-left block) ---
    # Causal attention among prepended summaries
    for i in range(num_prior_summaries):
        mask[i, : i + 1] = 0.0

    # --- Region 2: Token-to-summary (left columns, bottom rows) ---
    # All tokens can attend to all summary tokens
    if num_prior_summaries > 0:
        mask[num_prior_summaries:, :num_prior_summaries] = 0.0

    # --- Region 3: Token-to-token (bottom-right block) ---
    # Causal within-step, blocked across steps
    step_of = assign_tokens_to_steps(seq_len, boundary_positions)

    for i in range(seq_len):
        for j in range(i + 1):  # causal: only attend to positions <= i
            if step_of[j] == step_of[i]:
                # Same step: allow causal attention
                mask[num_prior_summaries + i, num_prior_summaries + j] = 0.0
            # Different step: remains -inf (blocked)

    # Expand to 4D: (batch, 1, total_len, total_len)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)


def build_cct_attention_mask_fast(
    seq_len: int,
    boundary_positions: list[int],
    num_prior_summaries: int,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Vectorized version of build_cct_attention_mask (no Python loops).

    Same semantics as build_cct_attention_mask but uses tensor ops
    for better performance on GPU with long sequences.
    """
    total_len = num_prior_summaries + seq_len
    mask = torch.full((total_len, total_len), float("-inf"), device=device)

    # Summary-to-summary: causal
    # mask[i, j] = 0 when j <= i (summary i attends to summaries 0..i)
    if num_prior_summaries > 0:
        s_idx = torch.arange(num_prior_summaries, device=device)
        # For result[i, j]: unsqueeze(1) broadcasts i along rows, unsqueeze(0) broadcasts j along cols
        # s_idx.unsqueeze(0) gives j values, s_idx.unsqueeze(1) gives i values
        # We want: j <= i
        summary_causal = s_idx.unsqueeze(0) <= s_idx.unsqueeze(1)
        mask[:num_prior_summaries, :num_prior_summaries] = torch.where(
            summary_causal, 0.0, float("-inf")
        )

    # Token-to-summary: all tokens attend to all summaries
    if num_prior_summaries > 0:
        mask[num_prior_summaries:, :num_prior_summaries] = 0.0

    # Token-to-token: causal within-step, blocked across steps
    # Build step assignment vector using the same logic as assign_tokens_to_steps
    step_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    if boundary_positions:
        boundary_set = set(boundary_positions)
        current_step = 0
        ids = []
        for pos in range(seq_len):
            ids.append(current_step)
            if pos in boundary_set:
                current_step += 1
        step_ids = torch.tensor(ids, dtype=torch.long, device=device)

    # same_step[i, j] = True iff token i and j are in the same step
    # unsqueeze(0) -> j values (cols), unsqueeze(1) -> i values (rows)
    same_step = step_ids.unsqueeze(0) == step_ids.unsqueeze(1)  # (seq, seq)

    # causal[i, j] = True iff j <= i (token i can attend to token j)
    t_idx = torch.arange(seq_len, device=device)
    causal = t_idx.unsqueeze(0) <= t_idx.unsqueeze(1)  # j <= i

    attend = same_step & causal
    mask[num_prior_summaries:, num_prior_summaries:] = torch.where(
        attend, 0.0, float("-inf")
    )

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
