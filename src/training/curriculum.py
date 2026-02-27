"""
4-Phase Commitment Probability Curriculum

Controls the probability that commitment is enforced at each step boundary.
When NOT enforced: standard autoregressive training (no mask, no gradient isolation).
When enforced: full CCT constraints apply.

Phase 1 (0-10%):   Awareness — p_commit = 0 (standard training)
Phase 2 (10-50%):  Stochastic — p_commit linearly 0.1 -> 0.5
Phase 3 (50-80%):  Majority — p_commit linearly 0.5 -> 0.9
Phase 4 (80-100%): Full — p_commit = 1.0

Noise annealing (optional): σ decreases over training, increasing the
information channel capacity through the commitment bottleneck.
  Early: high σ → low bitrate → model encodes only critical info
  Late:  low σ → high bitrate → model encodes everything it needs

See CCT_IMPLEMENTATION_GUIDE.md Section 1.5.
"""


class CommitmentCurriculum:
    """Schedules commitment probability, loss weights, and noise level."""

    def __init__(
        self,
        total_steps: int,
        phase1_end: float = 0.10,
        phase2_end: float = 0.50,
        phase3_end: float = 0.80,
        delta_start: float = 0.0,
        delta_max: float = 0.2,
        delta_taper: bool = False,
        gamma_max: float = 0.3,
        alpha_end: float = 0.5,
        beta_max: float = 0.5,
        aux_std_weight: float = 0.0,
        noise_sigma_start: float = 0.0,
        noise_sigma_end: float = 0.0,
    ):
        self.total = total_steps
        self.p1 = phase1_end
        self.p2 = phase2_end
        self.p3 = phase3_end
        self.delta_start = delta_start
        self.delta_max = delta_max
        self.delta_taper = delta_taper
        self.gamma_max = gamma_max
        self.alpha_end = alpha_end
        self.beta_max = beta_max
        self.aux_std_weight = aux_std_weight
        self.noise_sigma_start = noise_sigma_start
        self.noise_sigma_end = noise_sigma_end

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
        delta: sufficiency loss weight (increases, optionally tapers to 0)
        """
        progress = current_step / self.total

        # Smooth transition from standard training to commitment training
        # Ramps from 0 at phase1_end to 1 at phase3_end
        commitment_weight = max(0.0, min(1.0, (progress - self.p1) / (self.p3 - self.p1)))

        # Delta (L_suf) weight with optional tapering
        if self.delta_taper:
            # Ramp up during phase 2 (p1 → p2), then taper to 0 during phase 3 (p2 → p3)
            if progress <= self.p1:
                delta = self.delta_start
            elif progress <= self.p2:
                # Phase 2: ramp from delta_start to delta_max
                phase2_progress = (progress - self.p1) / (self.p2 - self.p1)
                delta = self.delta_start + (self.delta_max - self.delta_start) * phase2_progress
            elif progress <= self.p3:
                # Phase 3: taper from delta_max to 0
                phase3_progress = (progress - self.p2) / (self.p3 - self.p2)
                delta = self.delta_max * (1.0 - phase3_progress)
            else:
                # Phase 4: delta = 0
                delta = 0.0
        else:
            delta = self.delta_start + (self.delta_max - self.delta_start) * commitment_weight

        # Base alpha decreases as commitment ramps up
        alpha = 1.0 - (1.0 - self.alpha_end) * commitment_weight
        # aux_std_weight: additional L_std boost during phase 3+ to prevent base model drift
        if self.aux_std_weight > 0 and progress >= self.p2:
            alpha += self.aux_std_weight

        return {
            "alpha": alpha,
            "beta": self.beta_max * commitment_weight,
            "gamma": self.gamma_max * commitment_weight,
            "delta": delta,
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

    def get_noise_sigma(self, current_step: int) -> float:
        """Returns noise injection σ for current training step.

        Anneals linearly from noise_sigma_start to noise_sigma_end.
        Returns 0.0 if noise injection is not configured.
        """
        if self.noise_sigma_start == 0.0 and self.noise_sigma_end == 0.0:
            return 0.0
        progress = current_step / self.total
        return self.noise_sigma_start + (self.noise_sigma_end - self.noise_sigma_start) * progress
