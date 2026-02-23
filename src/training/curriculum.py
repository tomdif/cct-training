"""
4-Phase Commitment Probability Curriculum

Controls the probability that commitment is enforced at each step boundary.
When NOT enforced: standard autoregressive training (no mask, no gradient isolation).
When enforced: full CCT constraints apply.

Phase 1 (0-10%):   Awareness — p_commit = 0 (standard training)
Phase 2 (10-50%):  Stochastic — p_commit linearly 0.1 -> 0.5
Phase 3 (50-80%):  Majority — p_commit linearly 0.5 -> 0.9
Phase 4 (80-100%): Full — p_commit = 1.0

See CCT_IMPLEMENTATION_GUIDE.md Section 1.5.
"""


class CommitmentCurriculum:
    """Schedules commitment probability and loss weights across training."""

    def __init__(
        self,
        total_steps: int,
        phase1_end: float = 0.10,
        phase2_end: float = 0.50,
        phase3_end: float = 0.80,
        delta_start: float = 0.0,
    ):
        self.total = total_steps
        self.p1 = phase1_end
        self.p2 = phase2_end
        self.p3 = phase3_end
        self.delta_start = delta_start

    def get_commitment_probability(self, current_step: int) -> float:
        """Returns p_commit for current training step."""
        progress = current_step / self.total

        if progress < self.p1:
            # Phase 1: Awareness — no commitment enforcement
            return 0.0
        elif progress < self.p2:
            # Phase 2: Stochastic — p_commit linearly 0.1 -> 0.5
            phase_progress = (progress - self.p1) / (self.p2 - self.p1)
            return 0.1 + phase_progress * 0.4
        elif progress < self.p3:
            # Phase 3: Majority — p_commit linearly 0.5 -> 0.9
            phase_progress = (progress - self.p2) / (self.p3 - self.p2)
            return 0.5 + phase_progress * 0.4
        else:
            # Phase 4: Full Commitment
            return 1.0

    def get_loss_weights(self, current_step: int) -> dict[str, float]:
        """Returns loss component weights for current training step.

        alpha: standard autoregressive loss weight (decreases)
        beta:  per-step validity loss weight (increases)
        gamma: conclusion-from-premises loss weight (increases)
        delta: sufficiency loss weight (increases)
        """
        progress = current_step / self.total

        # Smooth transition from standard training to commitment training
        # Ramps from 0 at phase1_end to 1 at phase3_end
        commitment_weight = max(0.0, min(1.0, (progress - self.p1) / (self.p3 - self.p1)))

        return {
            "alpha": 1.0 - 0.5 * commitment_weight,  # 1.0 -> 0.5
            "beta": 0.5 * commitment_weight,  # 0.0 -> 0.5
            "gamma": 0.3 * commitment_weight,  # 0.0 -> 0.3
            "delta": self.delta_start + (0.2 - self.delta_start) * commitment_weight,
        }

    def get_phase(self, current_step: int) -> int:
        """Returns current phase number (1-4)."""
        progress = current_step / self.total
        if progress < self.p1:
            return 1
        elif progress < self.p2:
            return 2
        elif progress < self.p3:
            return 3
        else:
            return 4
