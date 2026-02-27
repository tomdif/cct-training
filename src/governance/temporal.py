"""Timestamp anchoring (system clock, RFC 3161 stub)."""

import hashlib
import time

from src.governance.proof import TemporalAnchor


def create_system_clock_anchor(chain_hash: bytes) -> TemporalAnchor:
    """Create a temporal anchor using the system clock."""
    timestamp = time.time()
    anchor_hash = hashlib.sha256(
        chain_hash + b"system_clock" + str(timestamp).encode()
    ).digest()
    return TemporalAnchor(
        chain_hash=chain_hash,
        anchor_type="system_clock",
        anchor_timestamp=timestamp,
        anchor_proof=b"",
        anchor_hash=anchor_hash,
    )


def create_rfc3161_anchor(chain_hash: bytes) -> TemporalAnchor:
    """
    Create a temporal anchor using RFC 3161 Timestamp Authority.
    Stub implementation — in production, this would call an external TSA.
    """
    timestamp = time.time()
    # In production: send chain_hash to TSA, get signed timestamp response
    stub_proof = hashlib.sha256(
        b"rfc3161_stub:" + chain_hash + str(timestamp).encode()
    ).digest()
    anchor_hash = hashlib.sha256(
        chain_hash + b"rfc3161" + str(timestamp).encode()
    ).digest()
    return TemporalAnchor(
        chain_hash=chain_hash,
        anchor_type="rfc3161",
        anchor_timestamp=timestamp,
        anchor_proof=stub_proof,
        anchor_hash=anchor_hash,
    )
