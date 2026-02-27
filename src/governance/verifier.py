"""
Independent verifier that takes a GovernanceProof and checks everything.
This is what a regulator or auditor would run.
"""

import hashlib
import json


class VerificationResult:
    def __init__(self):
        self.checks: list[dict] = []
        self.passed: bool = True
        self.summary: str = ""

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Verification Result: {status}"]
        lines.append(f"  Total checks: {len(self.checks)}")
        lines.append(f"  Passed: {sum(1 for c in self.checks if c['passed'])}")
        lines.append(f"  Failed: {sum(1 for c in self.checks if not c['passed'])}")
        for c in self.checks:
            mark = "+" if c["passed"] else "X"
            lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
        return "\n".join(lines)


def verify_governance_proof(proof) -> VerificationResult:
    """
    Complete independent verification of a GovernanceProof.

    Checks:
    1. Chain integrity (all hashes recompute correctly)
    2. Chain linkage (each step references prior step's hash)
    3. Timestamp monotonicity (non-decreasing)
    4. Genesis validity (first step's prior_hash matches genesis)
    5. Final hash matches declared final_chain_hash
    6. [Tier 3] All constraint evaluations passed
    7. [Tier 3] Temporal anchors reference valid chain hashes
    8. [Tier 3] Provenance tree roots bound to correct chain hashes
    9. [Tier 3] Provenance timestamps precede step timestamps
    """
    result = VerificationResult()
    steps = proof.steps

    if not steps:
        result.add_check("non_empty", False, "No steps in proof")
        return result

    result.add_check("non_empty", True, f"{len(steps)} steps")

    # 1-2. Chain integrity and linkage
    for i, step in enumerate(steps):
        metadata_bytes = json.dumps(step.metadata, sort_keys=True).encode()
        expected = hashlib.sha256(
            step.prior_chain_hash + step.summary_hash + metadata_bytes
        ).digest()

        hash_ok = (expected == step.chain_hash)
        result.add_check(
            f"chain_hash_step_{step.step_index}",
            hash_ok,
            "recomputed matches" if hash_ok else "MISMATCH"
        )

        if i > 0:
            link_ok = (step.prior_chain_hash == steps[i - 1].chain_hash)
            result.add_check(
                f"linkage_step_{step.step_index}",
                link_ok,
                "links to prior" if link_ok else "BROKEN LINK"
            )

    # 3. Timestamp monotonicity
    monotonic = all(
        steps[i].timestamp <= steps[i + 1].timestamp
        for i in range(len(steps) - 1)
    )
    result.add_check(
        "timestamp_monotonicity", monotonic,
        "all timestamps non-decreasing" if monotonic else "timestamps out of order"
    )

    # 4. Genesis
    result.add_check(
        "genesis_hash",
        steps[0].prior_chain_hash == proof.genesis_hash,
        "genesis matches" if steps[0].prior_chain_hash == proof.genesis_hash
        else "genesis mismatch"
    )

    # 5. Final hash
    result.add_check(
        "final_hash",
        steps[-1].chain_hash == proof.final_chain_hash,
        "final matches" if steps[-1].chain_hash == proof.final_chain_hash
        else "final mismatch"
    )

    # 6. Tier 3: constraint compliance
    if proof.tier == 3 and proof.constraint_evals:
        all_passed = all(
            all(r.passed for r in ce.results)
            for ce in proof.constraint_evals
        )
        result.add_check(
            "constraint_compliance",
            all_passed,
            f"all {len(proof.constraint_evals)} evals passed" if all_passed
            else "constraint violations detected"
        )

        for ce in proof.constraint_evals:
            for r in ce.results:
                if not r.passed:
                    result.add_check(
                        f"constraint_{r.constraint_id}_step_{ce.step_index}",
                        False,
                        r.detail
                    )

    # 7. Tier 3: temporal anchors reference valid chain hashes
    if proof.tier == 3 and proof.temporal_anchors:
        valid_hashes = {s.chain_hash for s in steps}
        for anchor in proof.temporal_anchors:
            ref_ok = anchor.chain_hash in valid_hashes
            result.add_check(
                f"temporal_anchor_{anchor.anchor_type}_step",
                ref_ok,
                "references valid chain hash" if ref_ok else "dangling reference"
            )

    # 8-9. Tier 3: provenance
    if proof.tier == 3 and proof.provenance_records:
        valid_hashes = {s.chain_hash for s in steps}
        step_times = {s.chain_hash: s.timestamp for s in steps}

        for prov in proof.provenance_records:
            bound_ok = prov.bound_to_chain_hash in valid_hashes
            result.add_check(
                f"provenance_binding_step_{prov.step_index}",
                bound_ok,
                "bound to valid chain hash" if bound_ok else "dangling"
            )

            if bound_ok:
                step_time = step_times[prov.bound_to_chain_hash]
                for src in prov.sources:
                    time_ok = src.data_timestamp <= step_time
                    if not time_ok:
                        result.add_check(
                            f"provenance_time_{prov.step_index}_{src.source_id}",
                            False,
                            f"source timestamp {src.data_timestamp} > "
                            f"step timestamp {step_time}"
                        )

    result.summary = (
        f"{'PASS' if result.passed else 'FAIL'}: "
        f"{sum(1 for c in result.checks if c['passed'])}/{len(result.checks)} checks"
    )

    return result
