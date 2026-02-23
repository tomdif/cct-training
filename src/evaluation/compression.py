"""
Memory Compression Measurement

Standard model at context length L:
  KV cache = n_layers * 2 * L * d_head * n_heads * sizeof(dtype)
           = n_layers * 2 * L * d_model * sizeof(dtype)

CCT model at same context length L with step_length S:
  Summary buffer = (L / S) * d_summary * sizeof(float32)
  Local KV cache = n_layers * 2 * S * d_model * sizeof(dtype)  (BOUNDED by step length)

Compression ratio = standard KV cache / (summary buffer + local KV cache)

See CCT_IMPLEMENTATION_GUIDE.md Section 2.4.
"""


def measure_compression(
    n_layers: int,
    d_model: int,
    d_summary: int,
    step_length: int,
    context_lengths: list[int],
    kv_dtype_bytes: int = 2,  # fp16/bf16
    summary_dtype_bytes: int = 4,  # fp32
) -> dict:
    """Compute theoretical compression ratios at various context lengths.

    Returns dict mapping context_length -> compression stats.
    """
    results = {}
    for L in context_lengths:
        # Standard KV cache: keys + values for all layers
        standard_kv_bytes = n_layers * 2 * L * d_model * kv_dtype_bytes

        # CCT: summary buffer + local (bounded) KV cache
        n_steps = L // step_length
        summary_buffer_bytes = n_steps * d_summary * summary_dtype_bytes
        local_kv_bytes = n_layers * 2 * step_length * d_model * kv_dtype_bytes
        cct_total_bytes = summary_buffer_bytes + local_kv_bytes

        ratio = standard_kv_bytes / cct_total_bytes if cct_total_bytes > 0 else float("inf")

        results[L] = {
            "standard_kv_mb": standard_kv_bytes / (1024 * 1024),
            "cct_summary_kb": summary_buffer_bytes / 1024,
            "cct_local_kv_mb": local_kv_bytes / (1024 * 1024),
            "cct_total_mb": cct_total_bytes / (1024 * 1024),
            "compression_ratio": ratio,
        }

    return results


def print_compression_table(results: dict, model_name: str = ""):
    """Pretty-print compression results."""
    header = f"Memory Compression — {model_name}" if model_name else "Memory Compression"
    print(f"\n{'=' * 70}")
    print(f"  {header}")
    print(f"{'=' * 70}")
    print(f"  {'Context':>10}  {'Std KV':>10}  {'CCT Total':>10}  {'Ratio':>8}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for L, stats in sorted(results.items()):
        std = f"{stats['standard_kv_mb']:.1f} MB"
        cct = f"{stats['cct_total_mb']:.2f} MB"
        ratio = f"{stats['compression_ratio']:.0f}x"
        print(f"  {L:>10,}  {std:>10}  {cct:>10}  {ratio:>8}")
    print()
