"""
Main CCT Training Loop

Integrates all components:
  attention masking + commitment head + gradient isolation +
  curriculum scheduler + loss function + evaluation

The key implementation insight (from the guide):
  The model sees the ENTIRE multi-step sequence in one forward pass.
  The attention mask enforces the step structure — no separate per-step
  forward passes needed for the main forward.

  For the validity loss, a SECOND forward pass is done where each step
  only sees prior commitment summaries (not any raw tokens from prior steps).
  This is what forces the model to put everything needed into the summaries.

Strategy A (Tier 1): Summary injection via hidden state replacement.
  1. First pass: standard causal attention over the whole sequence.
  2. At each STEP token position, extract the hidden state, compute the
     commitment summary via the commitment head, apply gradient isolation.
  3. Second pass: CCT-masked attention where STEP token positions carry
     the commitment summaries (up-projected to d_model), and cross-step
     token attention is blocked.
  4. Compute all loss components from the second pass.

See CCT_IMPLEMENTATION_GUIDE.md Section 1.7.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from src.model.commitment_head import CommitmentHead
from src.model.gradient_isolation import isolate_gradient
from src.model.cct_attention import build_cct_attention_mask_fast
from src.model.summary_buffer import SummaryBuffer
from src.training.curriculum import CommitmentCurriculum
from src.training.loss import compute_cct_loss, SufficiencyProbe
from src.training.data_pipeline import (
    get_step_boundary_positions,
    get_step_ranges,
)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    d_model: int = 768
    d_summary: int = 16
    d_bottleneck: int = 32
    step_length: int = 50

    # Training
    total_steps: int = 50000
    batch_size: int = 8
    seq_len: int = 512
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Curriculum
    phase1_end: float = 0.10
    phase2_end: float = 0.50
    phase3_end: float = 0.80
    delta_start: float = 0.0

    # Evaluation
    eval_interval: int = 5000
    log_interval: int = 100

    # Device
    device: str = "cuda"
    mixed_precision: str = "bf16"


class CCTTrainer:
    """Main CCT training loop.

    Handles the two-pass training strategy:
      Pass 1: Standard forward to get hidden states at STEP positions.
      Pass 2: CCT-masked forward with commitment summaries injected.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: TrainConfig,
        step_token_id: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.step_token_id = step_token_id
        self.device = torch.device(config.device)

        # CCT components
        self.commitment_head = CommitmentHead(
            d_model=config.d_model,
            d_summary=config.d_summary,
            d_bottleneck=config.d_bottleneck,
        ).to(self.device)

        self.sufficiency_probe = SufficiencyProbe(
            d_summary=config.d_summary,
            d_model=config.d_model,
        ).to(self.device)

        self.summary_buffer = SummaryBuffer(
            d_summary=config.d_summary,
            d_model=config.d_model,
            device=self.device,
        )

        self.curriculum = CommitmentCurriculum(
            total_steps=config.total_steps,
            phase1_end=config.phase1_end,
            phase2_end=config.phase2_end,
            phase3_end=config.phase3_end,
            delta_start=config.delta_start,
        )

        # Optimizer — includes commitment head and sufficiency probe params
        all_params = (
            list(model.parameters())
            + list(self.commitment_head.parameters())
            + list(self.sufficiency_probe.parameters())
            + list(self.summary_buffer.up_project.parameters())
        )
        self.optimizer = optim.AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_steps,
        )

        self.global_step = 0

    def _get_amp_context(self):
        """Get automatic mixed precision context."""
        if self.config.mixed_precision == "bf16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        elif self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        else:
            return torch.autocast(self.device.type, enabled=False)

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Execute one CCT training step.

        Args:
            input_ids: (batch, seq_len) token IDs with STEP tokens inserted.

        Returns:
            Dict of loss values for logging.
        """
        self.model.train()
        self.commitment_head.train()
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.to(self.device)

        # Get curriculum state
        p_commit = self.curriculum.get_commitment_probability(self.global_step)
        weights = self.curriculum.get_loss_weights(self.global_step)
        phase = self.curriculum.get_phase(self.global_step)

        # Phase 1: standard training only (no CCT)
        if phase == 1 or p_commit == 0.0:
            return self._standard_train_step(input_ids, weights)

        # Stochastic commitment: flip a coin per step boundary
        commit_this_step = (torch.rand(1).item() < p_commit)
        if not commit_this_step:
            return self._standard_train_step(input_ids, weights)

        # Full CCT training step
        return self._cct_train_step(input_ids, weights)

    def _standard_train_step(
        self, input_ids: torch.Tensor, weights: dict
    ) -> dict:
        """Standard autoregressive training step (no CCT constraints)."""
        labels = input_ids.clone()
        # Mask STEP tokens in labels (don't predict them)
        labels[labels == self.step_token_id] = -100
        # Shift for next-token prediction
        labels = labels[:, 1:]

        with self._get_amp_context():
            outputs = self.model(input_ids, output_hidden_states=False)
            logits = outputs.logits[:, :-1, :]  # align with labels
            loss_dict = compute_cct_loss(
                lm_logits=logits,
                labels=labels,
                weights={"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
            )

        loss = loss_dict["total"] / self.config.gradient_accumulation_steps
        loss.backward()

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        loss_dict["phase"] = self.curriculum.get_phase(self.global_step)
        loss_dict["p_commit"] = self.curriculum.get_commitment_probability(self.global_step)
        loss_dict["lr"] = self.scheduler.get_last_lr()[0]
        return loss_dict

    def _cct_train_step(
        self, input_ids: torch.Tensor, weights: dict
    ) -> dict:
        """Full CCT training step with commitment enforcement.

        Two-pass approach:
          Pass 1: Standard forward to extract hidden states at STEP positions,
                  compute commitment summaries.
          Pass 2: CCT-masked forward with summaries injected at STEP positions.
                  Compute all loss components.
        """
        batch_size, seq_len = input_ids.shape

        # Find STEP token positions (same across batch since data is uniformly annotated)
        # Use first batch element to find boundaries
        boundary_positions = get_step_boundary_positions(
            input_ids[0].tolist(), self.step_token_id
        )
        step_ranges = get_step_ranges(seq_len, boundary_positions)

        # === PASS 1: Standard forward to get hidden states ===
        with torch.no_grad():
            outputs_pass1 = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs_pass1.hidden_states[-1]  # last layer: (B, S, D)

        # Compute commitment summaries at each STEP boundary
        summaries = []           # detached — for injection into Pass 2
        summaries_live = []      # NOT detached — for sufficiency probe gradient
        step_hidden_means = []
        for start, end in step_ranges[:-1]:  # all steps except the last
            step_hidden = hidden_states[:, start:end, :]  # (B, step_len, D)
            # Cast to float32 for commitment head (model may output bf16)
            step_hidden_f32 = step_hidden.float()
            step_hidden_means.append(step_hidden_f32.mean(dim=1))  # (B, D)

            # Commitment head produces summary from this step
            summary = self.commitment_head(step_hidden_f32)  # (B, d_summary)
            summaries_live.append(summary)  # keep grad for probe
            summary_detached = isolate_gradient(summary)  # stop grad for injection
            summaries.append(summary_detached)

        # === PASS 2: CCT-masked forward ===
        # Inject commitment summaries by replacing STEP token embeddings
        # with up-projected summaries

        with self._get_amp_context():
            # Get the model's input embeddings
            if hasattr(self.model, "get_input_embeddings"):
                embed_layer = self.model.get_input_embeddings()
            else:
                embed_layer = self.model.gpt_neox.embed_in

            inputs_embeds = embed_layer(input_ids)  # (B, S, D)

            # Replace STEP token positions with up-projected summaries
            if summaries:
                for idx, bp in enumerate(boundary_positions):
                    if idx < len(summaries):
                        up_proj = self.summary_buffer.up_project(
                            summaries[idx]
                        )  # (B, D)
                        inputs_embeds[:, bp, :] = up_proj.to(inputs_embeds.dtype)

            # Build CCT attention mask
            cct_mask = build_cct_attention_mask_fast(
                seq_len=seq_len,
                boundary_positions=boundary_positions,
                num_prior_summaries=0,  # summaries injected in-place, not prepended
                device=self.device,
                batch_size=batch_size,
            )

            # Forward with CCT mask and modified embeddings
            outputs_pass2 = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=cct_mask,
                output_hidden_states=True,
            )
            logits = outputs_pass2.logits

            # Labels: next-token prediction, mask STEP tokens
            labels = input_ids.clone()
            labels[labels == self.step_token_id] = -100
            labels = labels[:, 1:]
            logits_shifted = logits[:, :-1, :]

            # Sufficiency probe inputs — use ALL steps with LIVE (non-detached) summaries
            # so gradient flows: L_sufficiency → probe → summary → commitment_head
            summary_for_probe = None
            hidden_mean_for_probe = None
            if summaries_live and step_hidden_means:
                # Stack all steps: (n_steps, B, dim) → (n_steps*B, dim)
                summary_for_probe = torch.cat(summaries_live, dim=0)
                hidden_mean_for_probe = torch.cat(step_hidden_means, dim=0)

            loss_dict = compute_cct_loss(
                lm_logits=logits_shifted,
                labels=labels,
                # Validity loss: use the same logits for now
                # (true validity would need a separate summary-only forward)
                validity_logits=logits_shifted if weights["beta"] > 0 else None,
                validity_labels=labels if weights["beta"] > 0 else None,
                # Sufficiency probe
                summary=summary_for_probe,
                step_hidden_mean=hidden_mean_for_probe,
                sufficiency_probe=self.sufficiency_probe if weights["delta"] > 0 else None,
                weights=weights,
            )

        loss = loss_dict["total"] / self.config.gradient_accumulation_steps
        loss.backward()

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            all_params = (
                list(self.model.parameters())
                + list(self.commitment_head.parameters())
                + list(self.sufficiency_probe.parameters())
                + list(self.summary_buffer.up_project.parameters())
            )
            nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        loss_dict["phase"] = self.curriculum.get_phase(self.global_step)
        loss_dict["p_commit"] = self.curriculum.get_commitment_probability(self.global_step)
        loss_dict["lr"] = self.scheduler.get_last_lr()[0]
        loss_dict["n_summaries"] = len(summaries)
        return loss_dict

    def save_checkpoint(self, path: str):
        """Save model and CCT component checkpoints."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "commitment_head_state_dict": self.commitment_head.state_dict(),
                "sufficiency_probe_state_dict": self.sufficiency_probe.state_dict(),
                "summary_up_project_state_dict": self.summary_buffer.up_project.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model and CCT component checkpoints."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.commitment_head.load_state_dict(ckpt["commitment_head_state_dict"])
        self.sufficiency_probe.load_state_dict(ckpt["sufficiency_probe_state_dict"])
        self.summary_buffer.up_project.load_state_dict(ckpt["summary_up_project_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
