"""GovernanceProof assembly, serialization, verification."""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DataSource:
    source_id: str               # "system_prompt", "user_message", "retrieval:doc_xyz"
    source_type: str             # "prompt", "retrieval", "tool_output", "memory"
    data_timestamp: float        # When the source data was created/published
    access_timestamp: float      # When the model accessed it
    content_hash: bytes          # Hash of the source content


@dataclass
class ProvenanceRecord:
    """Input Access Tree for one step."""
    step_index: int
    sources: list[DataSource]
    tree_root_hash: bytes        # Merkle root of the source list
    bound_to_chain_hash: bytes   # The chain_hash this provenance is bound to


@dataclass
class GovernanceStepRecord:
    """Produced at each step boundary under Tier 2+."""
    step_index: int
    summary_hash: bytes          # SHA-256(h_k as float32 bytes)
    chain_hash: bytes            # SHA-256(prior_chain_hash || summary_hash || metadata)
    prior_chain_hash: bytes      # Previous step's chain_hash (genesis for step 0)
    timestamp: float             # time.time() at commitment
    metadata: dict               # {"step_index": k, "tier": 2, "d_summary": 128, ...}


@dataclass
class ConstraintResult:
    constraint_id: str
    constraint_type: str         # "content_safety", "factual_grounding", "policy", etc.
    passed: bool
    score: float                 # 0.0-1.0 confidence/margin
    detail: str                  # Brief explanation


@dataclass
class ConstraintEvalRecord:
    """Produced by adversarial evaluator at each step under Tier 3."""
    step_index: int
    constraint_set_hash: bytes   # Hash of the constraint pack used
    results: list[ConstraintResult]
    evaluator_id: str            # Identifier for the evaluator model/version
    eval_timestamp: float
    eval_hash: bytes             # SHA-256(step_index || constraint_set_hash || results)


@dataclass
class TemporalAnchor:
    """Binds a chain_hash to an external time reference."""
    chain_hash: bytes
    anchor_type: str             # "system_clock", "rfc3161", "blockchain", "peer"
    anchor_timestamp: float
    anchor_proof: bytes          # External proof data (TSA response, tx hash, etc.)
    anchor_hash: bytes           # SHA-256(chain_hash || anchor_type || anchor_timestamp)


@dataclass
class GovernanceProof:
    """Complete governance artifact for one reasoning chain."""
    chain_id: str                # UUID for this chain
    model_id: str                # Model identifier + weight hash
    tier: int                    # 1, 2, or 3
    genesis_hash: bytes          # Hash of genesis block
    final_chain_hash: bytes      # Last step's chain_hash
    steps: list[GovernanceStepRecord]

    # Tier 3 only:
    constraint_evals: Optional[list[ConstraintEvalRecord]] = None
    temporal_anchors: Optional[list[TemporalAnchor]] = None
    provenance_records: Optional[list[ProvenanceRecord]] = None

    # Metadata:
    start_time: float = 0.0
    end_time: float = 0.0
    total_steps: int = 0
    config: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON (hex-encoding bytes fields)."""
        def _convert(obj):
            if isinstance(obj, bytes):
                return obj.hex()
            if isinstance(obj, (GovernanceStepRecord, ConstraintEvalRecord,
                                ConstraintResult, TemporalAnchor,
                                ProvenanceRecord, DataSource)):
                d = {}
                for k, v in obj.__dict__.items():
                    d[k] = _convert(v)
                return d
            if isinstance(obj, list):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        data = {
            "chain_id": self.chain_id,
            "model_id": self.model_id,
            "tier": self.tier,
            "genesis_hash": self.genesis_hash.hex(),
            "final_chain_hash": self.final_chain_hash.hex(),
            "steps": [_convert(s) for s in self.steps],
            "constraint_evals": [_convert(e) for e in self.constraint_evals] if self.constraint_evals else None,
            "temporal_anchors": [_convert(a) for a in self.temporal_anchors] if self.temporal_anchors else None,
            "provenance_records": [_convert(p) for p in self.provenance_records] if self.provenance_records else None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_steps": self.total_steps,
            "config": self.config,
        }
        return json.dumps(data, indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, json_str: str) -> "GovernanceProof":
        """Deserialize from JSON."""
        data = json.loads(json_str)

        def _bytes(hex_str):
            return bytes.fromhex(hex_str) if hex_str else b""

        steps = []
        for s in data["steps"]:
            steps.append(GovernanceStepRecord(
                step_index=s["step_index"],
                summary_hash=_bytes(s["summary_hash"]),
                chain_hash=_bytes(s["chain_hash"]),
                prior_chain_hash=_bytes(s["prior_chain_hash"]),
                timestamp=s["timestamp"],
                metadata=s["metadata"],
            ))

        constraint_evals = None
        if data.get("constraint_evals"):
            constraint_evals = []
            for e in data["constraint_evals"]:
                results = [
                    ConstraintResult(
                        constraint_id=r["constraint_id"],
                        constraint_type=r["constraint_type"],
                        passed=r["passed"],
                        score=r["score"],
                        detail=r["detail"],
                    )
                    for r in e["results"]
                ]
                constraint_evals.append(ConstraintEvalRecord(
                    step_index=e["step_index"],
                    constraint_set_hash=_bytes(e["constraint_set_hash"]),
                    results=results,
                    evaluator_id=e["evaluator_id"],
                    eval_timestamp=e["eval_timestamp"],
                    eval_hash=_bytes(e["eval_hash"]),
                ))

        temporal_anchors = None
        if data.get("temporal_anchors"):
            temporal_anchors = [
                TemporalAnchor(
                    chain_hash=_bytes(a["chain_hash"]),
                    anchor_type=a["anchor_type"],
                    anchor_timestamp=a["anchor_timestamp"],
                    anchor_proof=_bytes(a["anchor_proof"]),
                    anchor_hash=_bytes(a["anchor_hash"]),
                )
                for a in data["temporal_anchors"]
            ]

        provenance_records = None
        if data.get("provenance_records"):
            provenance_records = []
            for p in data["provenance_records"]:
                sources = [
                    DataSource(
                        source_id=s["source_id"],
                        source_type=s["source_type"],
                        data_timestamp=s["data_timestamp"],
                        access_timestamp=s["access_timestamp"],
                        content_hash=_bytes(s["content_hash"]),
                    )
                    for s in p["sources"]
                ]
                provenance_records.append(ProvenanceRecord(
                    step_index=p["step_index"],
                    sources=sources,
                    tree_root_hash=_bytes(p["tree_root_hash"]),
                    bound_to_chain_hash=_bytes(p["bound_to_chain_hash"]),
                ))

        return cls(
            chain_id=data["chain_id"],
            model_id=data["model_id"],
            tier=data["tier"],
            genesis_hash=_bytes(data["genesis_hash"]),
            final_chain_hash=_bytes(data["final_chain_hash"]),
            steps=steps,
            constraint_evals=constraint_evals,
            temporal_anchors=temporal_anchors,
            provenance_records=provenance_records,
            start_time=data["start_time"],
            end_time=data["end_time"],
            total_steps=data["total_steps"],
            config=data.get("config", {}),
        )

    def verify(self):
        """Self-verify the proof's internal consistency."""
        from src.governance.verifier import verify_governance_proof
        return verify_governance_proof(self)
