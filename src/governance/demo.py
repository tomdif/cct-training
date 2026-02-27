"""
End-to-end governance demo.

Usage:
    python -m src.governance.demo
    python -m src.governance.demo --tier 3
    python -m src.governance.demo --tier 2 --steps 20
"""

import argparse
import copy
import time

import torch

from src.governance.core import GovernanceEngine
from src.governance.constraints import ConstraintPack
from src.governance.verifier import verify_governance_proof


def run_demo(tier=2, n_steps=10, d_summary=128):
    """
    End-to-end governance demo.

    1. Run simulated CCT inference with governance enabled
    2. Verify the proof (should pass)
    3. Tamper with one step's record
    4. Verify again (should fail with specific error)
    5. Print timing/overhead analysis
    """

    # === Phase 1: Normal inference with governance ===
    engine = GovernanceEngine(
        tier=tier,
        config={"d_summary": d_summary, "step_length": 400}
    )

    constraint_pack = ConstraintPack() if tier == 3 else None
    engine.begin_chain(
        model_id="pythia-410m-cct-v6",
        constraint_pack=constraint_pack,
    )

    # Simulate step-by-step inference
    summaries = []
    t_inference_start = time.perf_counter()
    t_governance_total = 0.0

    for step_k in range(n_steps):
        # Simulate: commitment head produces a summary
        summary = torch.randn(d_summary)

        sources = None
        if tier == 3:
            sources = [
                {"id": "system_prompt", "type": "prompt",
                 "data_timestamp": engine.start_time - 1,
                 "content": "You are a helpful assistant."},
                {"id": f"user_turn_{step_k}", "type": "prompt",
                 "data_timestamp": engine.start_time,
                 "content": f"Step {step_k} input"},
            ]

        t0 = time.perf_counter()
        record = engine.commit_step(step_k, summary, sources=sources)
        t_governance_total += time.perf_counter() - t0
        summaries.append(summary)

    proof = engine.finalize()
    t_inference_total = time.perf_counter() - t_inference_start

    # === Phase 2: Verify (should pass) ===
    print("=" * 60)
    print(f"TIER {tier} GOVERNANCE DEMO")
    print("=" * 60)
    print(f"\nChain: {proof.chain_id}")
    print(f"Model: {proof.model_id}")
    print(f"Steps: {proof.total_steps}")
    print(f"Duration: {proof.end_time - proof.start_time:.3f}s")

    if tier == 1:
        print(f"\n  Tier 1: pass-through, no governance artifacts produced.")
        print(f"\n--- Overhead Analysis ---")
        print(f"  Proof size: 0 bytes")
        print(f"  Per-step overhead: N/A (tier 1 pass-through)")
        gov_ms = t_governance_total * 1000
        print(f"  Governance compute: {gov_ms:.2f}ms total, "
              f"{gov_ms / n_steps:.3f}ms/step")
        print()
        return proof

    print(f"\n--- Verification (original) ---")
    result = verify_governance_proof(proof)
    print(result)

    # === Phase 3: Tamper and re-verify ===
    if tier >= 2:
        # Tamper 1: Modify a summary hash
        if len(proof.steps) > 3:
            tampered = copy.deepcopy(proof)
            original_hash = tampered.steps[3].summary_hash
            tampered.steps[3].summary_hash = b"\x00" * 32

            print(f"\n--- Verification (tampered step 3 summary) ---")
            result2 = verify_governance_proof(tampered)
            print(result2)

        # Tamper 2: Swap two steps
        if len(proof.steps) > 5:
            tampered2 = copy.deepcopy(proof)
            tampered2.steps[4], tampered2.steps[5] = tampered2.steps[5], tampered2.steps[4]

            print(f"\n--- Verification (swapped steps 4 and 5) ---")
            result3 = verify_governance_proof(tampered2)
            print(result3)

        # Tamper 3: Break the chain by altering a chain_hash
        if len(proof.steps) > 2:
            tampered3 = copy.deepcopy(proof)
            tampered3.steps[1].chain_hash = b"\xff" * 32

            print(f"\n--- Verification (corrupted step 1 chain hash) ---")
            result4 = verify_governance_proof(tampered3)
            print(result4)

    # === Phase 4: Serialization round-trip ===
    print(f"\n--- Serialization Round-Trip ---")
    json_str = proof.to_json()
    proof_rt = proof.from_json(json_str)
    result_rt = verify_governance_proof(proof_rt)
    rt_ok = result_rt.passed
    print(f"  JSON size: {len(json_str):,} bytes ({len(json_str)/1024:.1f} KB)")
    print(f"  Round-trip verification: {'PASS' if rt_ok else 'FAIL'}")

    # === Phase 5: Overhead analysis ===
    print(f"\n--- Overhead Analysis ---")
    proof_size = _estimate_proof_size(proof)
    print(f"  Proof size: {proof_size:,} bytes ({proof_size / 1024:.1f} KB)")
    if proof.total_steps > 0:
        print(f"  Per-step overhead: {proof_size / proof.total_steps:.0f} bytes")
    else:
        print(f"  Per-step overhead: N/A (tier 1 pass-through)")
    gov_ms = t_governance_total * 1000
    print(f"  Governance compute: {gov_ms:.2f}ms total"
          + (f", {gov_ms / n_steps:.3f}ms/step" if n_steps > 0 else ""))
    if proof.total_steps > 0:
        print(f"  Chain computation: ~{proof.total_steps} SHA-256 ops")

    if tier == 3:
        print(f"  Constraint evals: {len(proof.constraint_evals)}")
        print(f"  Temporal anchors: {len(proof.temporal_anchors)}")
        print(f"  Provenance records: {len(proof.provenance_records)}")

    print()
    return proof


def _estimate_proof_size(proof):
    """Rough estimate of serialized proof size in bytes."""
    size = 0
    # Per step: 3 hashes (32 bytes each) + metadata (~100 bytes)
    size += len(proof.steps) * (32 * 3 + 100)

    if proof.constraint_evals:
        size += len(proof.constraint_evals) * 200

    if proof.temporal_anchors:
        size += len(proof.temporal_anchors) * 100

    if proof.provenance_records:
        for p in proof.provenance_records:
            size += 64 + len(p.sources) * 100

    return size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCP Governance Demo")
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--d-summary", type=int, default=128)
    args = parser.parse_args()

    print(f"Running governance demo: Tier {args.tier}, {args.steps} steps, "
          f"d_summary={args.d_summary}\n")

    # Run all tiers for comparison if tier=2 (default)
    if args.tier == 2 and args.steps == 10:
        for t in [1, 2, 3]:
            run_demo(tier=t, n_steps=args.steps, d_summary=args.d_summary)
    else:
        run_demo(tier=args.tier, n_steps=args.steps, d_summary=args.d_summary)
