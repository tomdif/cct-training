"""Constraint evaluator interface + built-in constraints."""

import hashlib
import time

from src.governance.proof import ConstraintResult, ConstraintEvalRecord


class ConstraintPack:
    """
    A collection of constraints to evaluate at each step.

    Built-in constraints for demo:
    - summary_norm: Summary L2 norm within expected range
    - summary_entropy: Summary values not collapsed to single mode

    Production constraints (pluggable):
    - content_safety: LLM-as-judge evaluating summary for policy violations
    - factual_grounding: Check claims against source documents
    - domain_rules: Custom business logic (financial limits, medical protocols, etc.)
    """

    def __init__(self, constraints: list = None):
        self.constraints = constraints or self._default_constraints()
        self.constraint_set_hash = self._compute_set_hash()

    def evaluate(self, step_index, summary_tensor, chain_hash) -> ConstraintEvalRecord:
        """Run all constraints against the step's summary."""
        results = []
        for c in self.constraints:
            result = c.check(step_index, summary_tensor)
            results.append(result)

        eval_hash = hashlib.sha256(
            str(step_index).encode()
            + self.constraint_set_hash
            + b"".join(
                (r.constraint_id + str(r.passed)).encode()
                for r in results
            )
        ).digest()

        return ConstraintEvalRecord(
            step_index=step_index,
            constraint_set_hash=self.constraint_set_hash,
            results=results,
            evaluator_id="builtin-v1",
            eval_timestamp=time.time(),
            eval_hash=eval_hash,
        )

    def _default_constraints(self):
        return [
            SummaryNormConstraint(),
            SummaryEntropyConstraint(),
        ]

    def _compute_set_hash(self):
        ids = sorted(c.constraint_id for c in self.constraints)
        return hashlib.sha256("|".join(ids).encode()).digest()


class SummaryNormConstraint:
    constraint_id = "summary_norm"
    constraint_type = "structural"

    def check(self, step_index, summary_tensor) -> ConstraintResult:
        import torch
        import math
        norm = torch.norm(summary_tensor.float()).item()
        d = summary_tensor.shape[-1]
        # Expected norm for d-dimensional vector: ~sqrt(d) for randn, varies for learned
        # Flag if norm is near-zero (dead summary) or extremely large (exploded)
        upper = 5.0 * math.sqrt(d)
        passed = 0.01 < norm < upper
        return ConstraintResult(
            constraint_id=self.constraint_id,
            constraint_type=self.constraint_type,
            passed=passed,
            score=min(norm / math.sqrt(d), 1.0),
            detail=f"L2 norm={norm:.4f} (d={d}, bound={upper:.1f})",
        )


class SummaryEntropyConstraint:
    constraint_id = "summary_entropy"
    constraint_type = "structural"

    def check(self, step_index, summary_tensor) -> ConstraintResult:
        import torch
        std = summary_tensor.float().std().item()
        passed = std > 0.01
        return ConstraintResult(
            constraint_id=self.constraint_id,
            constraint_type=self.constraint_type,
            passed=passed,
            score=min(std / 0.5, 1.0),
            detail=f"std={std:.4f}",
        )
