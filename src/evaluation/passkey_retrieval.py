"""
Passkey Retrieval Test

Insert a random passkey (e.g., "The passkey is 83719") at a random position
in a long context filled with padding text. Ask the model to recall the
passkey at the end.

For CCT: the passkey must survive the commitment summary bottleneck.
If the step containing the passkey produces a summary that encodes the
passkey, and a later step can extract it from that summary, sufficiency holds.

See CCT_IMPLEMENTATION_GUIDE.md Section 2.5.
"""

import random
import torch
import numpy as np


# Filler text for padding context
FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A stitch in time saves nine.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "Actions speak louder than words.",
    "Knowledge is power in the modern world.",
    "Practice makes perfect in every endeavor.",
    "Time and tide wait for no man.",
    "Fortune favors the bold and courageous.",
]


def generate_passkey_prompt(
    tokenizer,
    context_length: int,
    step_token_id: int | None = None,
    step_length: int = 50,
    seed: int | None = None,
) -> tuple[torch.Tensor, str, int]:
    """Generate a passkey retrieval test prompt.

    Args:
        tokenizer: The tokenizer.
        context_length: Target total token count.
        step_token_id: If provided, insert STEP tokens.
        step_length: Tokens per step (for STEP insertion).
        seed: Random seed for reproducibility.

    Returns:
        input_ids: (1, context_length) tensor.
        passkey: The passkey string (e.g., "83719").
        passkey_position: Token position where the passkey was inserted.
    """
    rng = random.Random(seed)

    # Generate random 5-digit passkey
    passkey = str(rng.randint(10000, 99999))
    passkey_sentence = f"The passkey is {passkey}. Remember this number."

    # Build filler text
    filler_text = " ".join(
        rng.choice(FILLER_SENTENCES) for _ in range(context_length // 5)
    )
    filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)

    # Encode passkey sentence
    passkey_tokens = tokenizer.encode(passkey_sentence, add_special_tokens=False)

    # Encode retrieval prompt
    retrieval_prompt = "What is the passkey? The passkey is"
    retrieval_tokens = tokenizer.encode(retrieval_prompt, add_special_tokens=False)

    # Calculate how many filler tokens we need
    total_filler_needed = context_length - len(passkey_tokens) - len(retrieval_tokens) - 5
    if total_filler_needed < 0:
        total_filler_needed = 0

    # Truncate or extend filler
    while len(filler_tokens) < total_filler_needed:
        filler_tokens.extend(
            tokenizer.encode(
                rng.choice(FILLER_SENTENCES), add_special_tokens=False
            )
        )
    filler_tokens = filler_tokens[:total_filler_needed]

    # Insert passkey at random position (10%-80% through the filler)
    insert_pos = rng.randint(
        len(filler_tokens) // 10,
        max(len(filler_tokens) // 10 + 1, int(len(filler_tokens) * 0.8)),
    )
    all_tokens = (
        filler_tokens[:insert_pos]
        + passkey_tokens
        + filler_tokens[insert_pos:]
        + retrieval_tokens
    )

    # Optionally insert STEP tokens
    if step_token_id is not None:
        annotated = []
        for i, tid in enumerate(all_tokens):
            annotated.append(tid)
            if (i + 1) % step_length == 0 and (i + 1) < len(all_tokens):
                annotated.append(step_token_id)
        all_tokens = annotated

    # Truncate to context_length
    all_tokens = all_tokens[:context_length]

    input_ids = torch.tensor([all_tokens], dtype=torch.long)
    return input_ids, passkey, insert_pos


def evaluate_passkey_retrieval(
    model,
    tokenizer,
    context_lengths: list[int],
    step_token_id: int | None = None,
    step_length: int = 50,
    n_trials: int = 10,
    device: torch.device = torch.device("cpu"),
    max_new_tokens: int = 10,
) -> dict:
    """Run passkey retrieval at multiple context lengths.

    Returns dict mapping context_length -> accuracy.
    """
    model.eval()
    results = {}

    for ctx_len in context_lengths:
        correct = 0
        for trial in range(n_trials):
            input_ids, passkey, _ = generate_passkey_prompt(
                tokenizer,
                context_length=ctx_len,
                step_token_id=step_token_id,
                step_length=step_length,
                seed=trial * 1000 + ctx_len,
            )
            input_ids = input_ids.to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

            # Decode only the new tokens
            new_tokens = generated[0, input_ids.shape[1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Check if passkey appears in the output
            if passkey in output_text:
                correct += 1

        accuracy = correct / n_trials
        results[ctx_len] = accuracy

    return results
