"""
Synthetic long-dependency data for recurrent commitment head training.

Entity repetition injection: copies marker spans from early steps to later
steps at multiple distances. The model must predict the repeated markers —
impossible without cross-step memory chaining through the recurrent head.

Multi-marker mode injects 2-3 markers per sequence at different hop distances:
  - Hop 2: basic survival (info persists one recurrent update)
  - Hop 3-4: minimal compounding
  - Hop 5-8: true long-range accumulation

Mixed with real web text at a configurable ratio (e.g., 30%).
"""

import random

import torch
from torch.utils.data import IterableDataset

from src.training.data_pipeline import annotate_step_boundaries


class SyntheticLongDepDataset(IterableDataset):
    """Streaming dataset with injected cross-step token repetitions.

    For each sequence:
    1. Tokenize web text normally
    2. Inject 1-3 marker spans at different hop distances
    3. Each marker is copied from step K to step K+d
    4. Annotate with STEP tokens and yield

    The repeated spans create dependencies that only a recurrent summary
    can exploit — the model needs to remember step K's content to predict
    step K+d's prefix.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        seq_len: int,
        step_token_id: int,
        step_length: int,
        min_distance: int = 1,
        max_distance: int = 8,
        marker_min: int = 8,
        marker_max: int = 15,
        multi_marker: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.step_token_id = step_token_id
        self.step_length = step_length
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.marker_min = marker_min
        self.marker_max = marker_max
        self.multi_marker = multi_marker

        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def _inject_marker(self, chunk, n_steps, used_targets):
        """Inject a single marker from source step to target step.

        Returns (source_step, target_step, distance) or None if no valid pair.
        """
        # Find valid source/target pairs avoiding already-used targets
        max_source = n_steps - self.min_distance - 1
        if max_source < 0:
            return None

        # Try a few times to find a non-colliding pair
        for _ in range(5):
            source_step = random.randint(0, max_source)
            max_dist = min(self.max_distance, n_steps - source_step - 1)
            if max_dist < self.min_distance:
                continue
            target_step = source_step + random.randint(self.min_distance, max_dist)
            if target_step >= n_steps or target_step in used_targets:
                continue

            # Extract marker from source step
            src_start = source_step * self.step_length
            marker_len = random.randint(self.marker_min, self.marker_max)
            marker_len = min(marker_len, self.step_length)
            marker_offset = random.randint(0, max(0, self.step_length - marker_len))
            marker = chunk[src_start + marker_offset : src_start + marker_offset + marker_len]

            # Inject at target step prefix
            tgt_start = target_step * self.step_length
            chunk[tgt_start : tgt_start + marker_len] = marker
            used_targets.add(target_step)
            return (source_step, target_step, target_step - source_step)

        return None

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]

                n_steps = len(chunk) // self.step_length
                if n_steps < self.min_distance + 2:
                    annotated = annotate_step_boundaries(
                        chunk, self.step_token_id, "fixed", self.step_length
                    )
                    annotated = annotated[:self.seq_len]
                    yield torch.tensor(annotated, dtype=torch.long)
                    continue

                used_targets = set()

                if self.multi_marker and n_steps >= 6:
                    # Multi-marker: inject 2-3 markers at different distances
                    # Prioritize different distance ranges
                    n_markers = min(3, n_steps // 3)

                    # Try short hop first (2-3), then medium (3-5), then long (5-8)
                    hop_ranges = [(2, 3), (3, 5), (5, 8)]
                    for i in range(n_markers):
                        lo, hi = hop_ranges[min(i, len(hop_ranges) - 1)]
                        old_min, old_max = self.min_distance, self.max_distance
                        self.min_distance = max(lo, 1)
                        self.max_distance = min(hi, n_steps - 1)
                        self._inject_marker(chunk, n_steps, used_targets)
                        self.min_distance, self.max_distance = old_min, old_max
                else:
                    # Single marker (original behavior)
                    self._inject_marker(chunk, n_steps, used_targets)

                annotated = annotate_step_boundaries(
                    chunk, self.step_token_id, "fixed", self.step_length
                )
                annotated = annotated[:self.seq_len]
                yield torch.tensor(annotated, dtype=torch.long)
