#!/usr/bin/env python3
"""
Sweep #1: Memory efficiency (vary max_bank_size and retrieval_k)
Sweep #3: Extreme length eval (16K, 32K, 64K tokens)

All eval-only — no training. Uses the 15K Run P checkpoint.
"""

import subprocess
import sys
import json
import os
from datetime import datetime

PYTHON = sys.executable
SCRIPT = "scripts/eval_retrieval.py"
CONFIG = "configs/tier3_410m_retrieval_P.yaml"
CHECKPOINT = "checkpoints-run-P-retrieval/cct-final.pt"
RESULTS_DIR = "./results-sweep-memory-length"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Skip conditions we don't need for sweep speed
SKIP = ["--skip-isolated"]  # isolated is slow and not needed for this sweep

def run_eval(name, seq_len, max_bank, retrieval_k, eval_batches=50):
    """Run one eval configuration and save results."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  seq_len={seq_len}, max_bank_size={max_bank}, retrieval_k={retrieval_k}")
    print(f"{'='*70}\n")

    cmd = [
        PYTHON, "-u", SCRIPT,
        "--config", CONFIG,
        "--cct-checkpoint", CHECKPOINT,
        "--seq-len", str(seq_len),
        "--max-bank-size", str(max_bank),
        "--retrieval-k", str(retrieval_k),
        "--eval-batches", str(eval_batches),
    ] + SKIP

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    # Save raw output
    outfile = os.path.join(RESULTS_DIR, f"{name}.txt")
    with open(outfile, "w") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"seq_len={seq_len}, max_bank_size={max_bank}, retrieval_k={retrieval_k}\n\n")
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"ERROR (exit {result.returncode}):")
        print(result.stderr[-1000:])

    return result.returncode == 0


def main():
    start = datetime.now()
    print(f"Starting sweep at {start.strftime('%H:%M:%S')}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Results: {RESULTS_DIR}")

    results = []

    # ===== SWEEP 1: Memory Efficiency =====
    # Baseline is M=8, k=2 (already have from Run P eval)
    # Test smaller banks to find minimum viable config

    memory_configs = [
        ("mem_M8_k2_2048",  2048, 8, 2),   # baseline for comparison
        ("mem_M4_k2_2048",  2048, 4, 2),   # half bank
        ("mem_M2_k1_2048",  2048, 2, 1),   # minimal bank
        ("mem_M1_k1_2048",  2048, 1, 1),   # absolute minimum
        ("mem_M4_k1_2048",  2048, 4, 1),   # single retrieval, medium bank
    ]

    print("\n" + "="*70)
    print("  SWEEP 1: MEMORY EFFICIENCY")
    print("="*70)

    for name, seq_len, max_bank, k in memory_configs:
        ok = run_eval(name, seq_len, max_bank, k)
        results.append({"name": name, "ok": ok})

    # ===== SWEEP 3: Extreme Length =====
    # Use baseline config (M=8, k=2) at increasing lengths

    length_configs = [
        ("len_16384_M8_k2",  16384, 8, 2),
        ("len_32768_M8_k2",  32768, 8, 2),
        ("len_65536_M8_k2",  65536, 8, 2),
    ]

    print("\n" + "="*70)
    print("  SWEEP 3: EXTREME LENGTH")
    print("="*70)

    for name, seq_len, max_bank, k in length_configs:
        # Fewer batches for longer seqs (memory + time)
        batches = 20 if seq_len <= 16384 else 10 if seq_len <= 32768 else 5
        ok = run_eval(name, seq_len, max_bank, k, eval_batches=batches)
        results.append({"name": name, "ok": ok})

    elapsed = datetime.now() - start
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — {elapsed}")
    print(f"{'='*70}")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['name']}")


if __name__ == "__main__":
    main()
