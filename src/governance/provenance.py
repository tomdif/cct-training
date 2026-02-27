"""Input Access Tree construction."""

import hashlib
import time

from src.governance.proof import DataSource, ProvenanceRecord
from src.governance.hash_chain import merkle_root


def build_provenance(step_index: int, sources: list[dict], chain_hash: bytes) -> ProvenanceRecord:
    """
    Build Input Access Tree from source list.

    Each source dict should have:
        id: str, type: str, content: str,
        data_timestamp: float (optional), access_timestamp: float (optional)
    """
    data_sources = []
    for s in sources:
        content_hash = hashlib.sha256(
            s.get("content", "").encode()
        ).digest()
        data_sources.append(DataSource(
            source_id=s["id"],
            source_type=s["type"],
            data_timestamp=s.get("data_timestamp", 0),
            access_timestamp=s.get("access_timestamp", time.time()),
            content_hash=content_hash,
        ))

    leaves = [s.content_hash for s in data_sources]
    tree_root = merkle_root(leaves)

    return ProvenanceRecord(
        step_index=step_index,
        sources=data_sources,
        tree_root_hash=tree_root,
        bound_to_chain_hash=chain_hash,
    )
