"""KV cache utilities for retrieval-augmented CCT training and evaluation.

Shared between src/training/trainer.py (training-time retrieval) and
scripts/eval_retrieval.py (eval-time retrieval).
"""

import torch
from typing import List, Optional, Tuple

KVTuples = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def _cache_to_tuples(cache, n_layers: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert any cache format to list of (key, value) tuples."""
    result = []
    # Newer transformers: .layers list with .keys/.values per layer
    if hasattr(cache, 'layers') and isinstance(cache.layers, list):
        for layer in cache.layers:
            result.append((layer.keys, layer.values))
        return result
    # Mid-version: .key_cache/.value_cache lists
    if hasattr(cache, 'key_cache'):
        for i in range(n_layers):
            result.append((cache.key_cache[i], cache.value_cache[i]))
        return result
    # Legacy: tuple of (key, value) per layer
    try:
        for i in range(n_layers):
            item = cache[i]
            result.append((item[0], item[1]))
        return result
    except (TypeError, IndexError, KeyError):
        pass
    raise ValueError(
        f"Cannot extract KV from {type(cache)}. "
        f"Attrs: {[a for a in dir(cache) if not a.startswith('__')]}"
    )


def _tuples_to_cache(kv_tuples):
    """Convert list/tuple of (key, value) back to DynamicCache for model input."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_tuples):
        cache.update(k, v, layer_idx)
    return cache


def extract_step_kv(output_cache, step_token_len: int, n_layers: int) -> KVTuples:
    """Extract KV for only the step's own tokens from model output cache.

    Returns a tuple of (key, value) per layer (plain tensors, no DynamicCache).
    """
    raw = _cache_to_tuples(output_cache, n_layers)
    result = []
    for k_full, v_full in raw:
        k_step = k_full[:, :, -step_token_len:, :].clone()
        v_step = v_full[:, :, -step_token_len:, :].clone()
        result.append((k_step, v_step))
    return tuple(result)


def merge_kv_caches(caches: List[KVTuples], n_layers: int) -> Optional[KVTuples]:
    """Concatenate multiple step KV caches along the sequence dimension.

    Input: list of tuple-of-(key,value) caches.
    Returns: tuple of (key, value) per layer.
    """
    if not caches:
        return None
    if len(caches) == 1:
        return caches[0]

    result = []
    for layer_idx in range(n_layers):
        keys = [c[layer_idx][0] for c in caches]
        values = [c[layer_idx][1] for c in caches]
        result.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    return tuple(result)
