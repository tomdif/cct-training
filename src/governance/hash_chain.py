"""SHA-256 chain operations, genesis, verification."""

import hashlib
import json


def compute_chain_hash(prior_hash: bytes, summary_hash: bytes, metadata: dict) -> bytes:
    """Compute H_k = SHA-256(H_{k-1} || summary_hash || canonical_metadata)."""
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
    return hashlib.sha256(prior_hash + summary_hash + metadata_bytes).digest()


def verify_chain(steps: list) -> tuple[bool, list[str]]:
    """
    Verify a chain of GovernanceStepRecords.
    Returns (valid, errors).
    """
    errors = []

    for i, step in enumerate(steps):
        # Recompute chain hash
        expected = compute_chain_hash(
            step.prior_chain_hash,
            step.summary_hash,
            step.metadata,
        )
        if expected != step.chain_hash:
            errors.append(f"Step {step.step_index}: chain hash mismatch")

        # Verify linkage
        if i > 0 and step.prior_chain_hash != steps[i - 1].chain_hash:
            errors.append(
                f"Step {step.step_index}: prior_chain_hash doesn't match "
                f"step {steps[i - 1].step_index}'s chain_hash"
            )

        # Verify monotonic timestamps
        if i > 0 and step.timestamp < steps[i - 1].timestamp:
            errors.append(
                f"Step {step.step_index}: timestamp {step.timestamp} < "
                f"prior step timestamp {steps[i - 1].timestamp}"
            )

    return (len(errors) == 0, errors)


def merkle_root(leaves: list[bytes]) -> bytes:
    """Compute Merkle root from leaf hashes."""
    if not leaves:
        return hashlib.sha256(b"empty").digest()
    if len(leaves) == 1:
        return leaves[0]

    # Pad to even
    if len(leaves) % 2 == 1:
        leaves = leaves + [leaves[-1]]

    parents = []
    for i in range(0, len(leaves), 2):
        combined = hashlib.sha256(leaves[i] + leaves[i + 1]).digest()
        parents.append(combined)

    return merkle_root(parents)
