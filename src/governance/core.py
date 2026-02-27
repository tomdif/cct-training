"""GovernanceEngine class, tier dispatch."""

import hashlib
import json
import time
import uuid
from typing import Optional

from src.governance.proof import (
    GovernanceProof,
    GovernanceStepRecord,
    TemporalAnchor,
)
from src.governance.hash_chain import compute_chain_hash
from src.governance.provenance import build_provenance


class GovernanceEngine:
    """
    Attaches to CCT inference loop. Called at each step boundary.

    Usage:
        engine = GovernanceEngine(tier=2, config={...})
        engine.begin_chain(model_id="pythia-410m-cct")

        for step_k in inference_loop:
            summary = commitment_head(hidden_states)
            engine.commit_step(step_k, summary)

        proof = engine.finalize()
        result = proof.verify()
    """

    def __init__(self, tier: int = 2, config: dict = None):
        assert tier in (1, 2, 3)
        self.tier = tier
        self.config = config or {}
        self.chain_id: str = ""
        self.model_id: str = ""
        self.steps: list[GovernanceStepRecord] = []
        self.constraint_evals = []
        self.temporal_anchors = []
        self.provenance_records = []
        self.start_time: float = 0
        self._prior_chain_hash: bytes = b""
        self._constraint_pack = None

    def begin_chain(self, model_id: str, constraint_pack=None):
        """Initialize a new governance chain."""
        self.chain_id = str(uuid.uuid4())
        self.model_id = model_id
        self.start_time = time.time()
        self.steps = []
        self.constraint_evals = []
        self.temporal_anchors = []
        self.provenance_records = []
        self._constraint_pack = constraint_pack

        # Genesis block
        genesis_data = json.dumps({
            "chain_id": self.chain_id,
            "model_id": self.model_id,
            "tier": self.tier,
            "start_time": self.start_time,
            "config": self.config
        }, sort_keys=True).encode()
        self._prior_chain_hash = hashlib.sha256(genesis_data).digest()

    def commit_step(
        self,
        step_index: int,
        summary_tensor,           # torch.Tensor, shape (d_summary,)
        sources: list = None,     # For Tier 3 provenance
    ) -> Optional[GovernanceStepRecord]:
        """
        Process one step boundary. Called after commitment head produces summary.

        Tier 1: Returns None (no-op)
        Tier 2: Hashes into chain, returns GovernanceStepRecord
        Tier 3: Hash + constraint eval + temporal anchor + provenance
        """
        if self.tier == 1:
            return None

        timestamp = time.time()

        # 1. Hash the summary
        summary_bytes = summary_tensor.detach().cpu().float().numpy().tobytes()
        summary_hash = hashlib.sha256(summary_bytes).digest()

        # 2. Build metadata
        metadata = {
            "step_index": step_index,
            "tier": self.tier,
            "d_summary": int(summary_tensor.shape[-1]),
            "timestamp": timestamp,
        }

        # 3. Compute chain hash
        chain_hash = compute_chain_hash(
            self._prior_chain_hash, summary_hash, metadata
        )

        # 4. Create step record
        record = GovernanceStepRecord(
            step_index=step_index,
            summary_hash=summary_hash,
            chain_hash=chain_hash,
            prior_chain_hash=self._prior_chain_hash,
            timestamp=timestamp,
            metadata=metadata,
        )
        self.steps.append(record)
        self._prior_chain_hash = chain_hash

        # 5. Tier 3: constraint evaluation
        if self.tier == 3 and self._constraint_pack is not None:
            eval_record = self._constraint_pack.evaluate(
                step_index=step_index,
                summary_tensor=summary_tensor,
                chain_hash=chain_hash,
            )
            self.constraint_evals.append(eval_record)

        # 6. Tier 3: temporal anchoring
        if self.tier == 3:
            anchor = TemporalAnchor(
                chain_hash=chain_hash,
                anchor_type="system_clock",
                anchor_timestamp=timestamp,
                anchor_proof=b"",
                anchor_hash=hashlib.sha256(
                    chain_hash + b"system_clock" +
                    str(timestamp).encode()
                ).digest(),
            )
            self.temporal_anchors.append(anchor)

        # 7. Tier 3: provenance
        if self.tier == 3 and sources is not None:
            prov = build_provenance(step_index, sources, chain_hash)
            self.provenance_records.append(prov)

        return record

    def finalize(self) -> GovernanceProof:
        """Finalize the chain and produce the complete GovernanceProof."""
        return GovernanceProof(
            chain_id=self.chain_id,
            model_id=self.model_id,
            tier=self.tier,
            genesis_hash=self.steps[0].prior_chain_hash if self.steps else b"",
            final_chain_hash=self._prior_chain_hash,
            steps=self.steps,
            constraint_evals=self.constraint_evals if self.tier == 3 else None,
            temporal_anchors=self.temporal_anchors if self.tier == 3 else None,
            provenance_records=self.provenance_records if self.tier == 3 else None,
            start_time=self.start_time,
            end_time=time.time(),
            total_steps=len(self.steps),
            config=self.config,
        )
