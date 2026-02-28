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
from src.model.summary_conditioner import SummaryConditioner
from src.model.summary_adapter import SummaryAdapter
from src.model.summary_logit_bias import SummaryLogitBias
from src.model.pseudo_token_decoder import PseudoTokenDecoder
from src.model.model_utils import get_model_layers
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
    delta_max: float = 0.2
    delta_taper: bool = False
    gamma_max: float = 0.3
    alpha_end: float = 0.5
    beta_max: float = 0.5
    aux_std_weight: float = 0.0

    # Commitment head options
    use_tanh: bool = True
    use_l2_norm: bool = True
    noise_injection: bool = False
    noise_sigma_start: float = 0.0
    noise_sigma_end: float = 0.0

    # Decoder options
    decoder_type: str = "linear"
    decoder_bottleneck: int | None = None

    # Multi-token summaries
    n_summary_tokens: int = 1  # K: number of summary tokens per step

    # L_conclusion_from_premises
    conclusion_n_tokens: int = 10  # N target tokens to predict from summaries only

    # Summary attention bias — learnable per-head-per-layer bias on summary positions
    summary_attn_bias: bool = False
    summary_attn_bias_init: float = 5.0

    # Summary conditioning — FiLM + gated residual injection (bypasses attention)
    summary_conditioning: bool = False

    # Summary adapter — gated cross-attention (for frozen-model training)
    summary_adapter: bool = False
    adapter_d: int = 64  # adapter bottleneck dimension
    freeze_base_model: bool = False  # freeze all base model weights
    summary_logit_bias: bool = False
    logit_bias_hidden_dim: int = 256

    # KV-cache prefix delivery — generate per-layer K/V from summaries
    use_kv_prefix: bool = False
    kv_prefix_hidden: int = 512  # hidden dim in prefix generator MLPs

    # Pseudo-token decoder — expand summary into multiple attention-compatible tokens
    use_pseudo_tokens: bool = False
    n_pseudo_tokens: int = 8      # number of pseudo-tokens per summary
    pseudo_decoder_hidden: int = 512  # hidden dim in decoder MLP

    # LoRA — lightweight adaptation of base model attention
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list = field(default_factory=lambda: ["query_key_value"])
    lora_layers_min: int = 12   # first layer to adapt (0-indexed)
    lora_layers_max: int = 23   # last layer to adapt (inclusive)
    lora_lr: float = 2e-5       # separate LR for LoRA params

    # Gradient isolation
    gradient_isolation: str = "full_detach"  # "full_detach" | "scaled" | "none"
    gradient_scale: float = 0.1              # gradient scaling when isolation == "scaled"

    # Summary injection
    summary_injection_mode: str = "replace"  # "replace" | "additive"

    # Training mode
    training_mode: str = "two_pass"  # "two_pass" | "sequential"

    # Recurrent commitment head
    recurrent_commitment: bool = False
    last_summary_only: bool = False  # Only pass latest summary to logit bias

    # Multi-hop loss: gradient flows through full GRU chain
    multi_hop_loss: bool = False
    multi_hop_weight: float = 0.25  # weight in overall loss

    # Synthetic data mixing ratio (0 = disabled)
    synthetic_data_ratio: float = 0.0

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
            use_tanh=config.use_tanh,
            use_l2_norm=config.use_l2_norm,
            noise_injection=config.noise_injection,
            n_summary_tokens=config.n_summary_tokens,
            recurrent=config.recurrent_commitment,
        ).to(self.device)

        # Sufficiency probe input: K * d_summary (flattened multi-token summary)
        self.sufficiency_probe = SufficiencyProbe(
            d_summary=config.d_summary * config.n_summary_tokens,
            d_model=config.d_model,
        ).to(self.device)

        self.summary_buffer = SummaryBuffer(
            d_summary=config.d_summary,
            d_model=config.d_model,
            device=self.device,
            decoder_type=config.decoder_type,
            decoder_bottleneck=config.decoder_bottleneck,
        )

        self.curriculum = CommitmentCurriculum(
            total_steps=config.total_steps,
            phase1_end=config.phase1_end,
            phase2_end=config.phase2_end,
            phase3_end=config.phase3_end,
            delta_start=config.delta_start,
            delta_max=config.delta_max,
            delta_taper=config.delta_taper,
            gamma_max=config.gamma_max,
            alpha_end=config.alpha_end,
            beta_max=config.beta_max,
            aux_std_weight=config.aux_std_weight,
            noise_sigma_start=config.noise_sigma_start,
            noise_sigma_end=config.noise_sigma_end,
        )

        # Summary attention bias — per-head-per-layer learnable bias
        # Forces attention to summary tokens by adding a positive bias
        # to attention logits at summary positions.
        self.summary_attn_bias_param = None
        self._summary_n_tokens = 0  # Set before each forward, read by hooks
        if config.summary_attn_bias:
            n_heads = model.config.num_attention_heads
            n_layers = model.config.num_hidden_layers
            self.summary_attn_bias_param = nn.Parameter(
                torch.full((n_layers, n_heads), config.summary_attn_bias_init,
                           device=self.device)
            )
            self._bias_hooks = []
            for i in range(n_layers):
                hook = get_model_layers(model)[i].register_forward_pre_hook(
                    self._make_bias_hook(i), with_kwargs=True,
                )
                self._bias_hooks.append(hook)

        # Summary conditioner — FiLM + gated residual injection
        self.summary_conditioner = None
        if config.summary_conditioning:
            n_layers = model.config.num_hidden_layers
            self.summary_conditioner = SummaryConditioner(
                d_summary=config.d_summary,
                d_model=config.d_model,
                n_layers=n_layers,
                device=self.device,
            )
            self.summary_conditioner.register_hooks(model)
            n_cond_params = sum(p.numel() for p in self.summary_conditioner.parameters())
            print(f"  Summary conditioner: {n_cond_params:,} params ({n_layers} layers)")

        # Summary adapter — gated cross-attention from summary bank
        self.summary_adapter = None
        if config.summary_adapter:
            n_layers = model.config.num_hidden_layers
            self.summary_adapter = SummaryAdapter(
                d_model=config.d_model,
                d_summary=config.d_summary,
                n_layers=n_layers,
                d_adapter=config.adapter_d,
                device=self.device,
            )
            self.summary_adapter.register_hooks(model)
            print(f"  Summary adapter: {self.summary_adapter.param_count():,} params "
                  f"({n_layers} layers, d_adapter={config.adapter_d})")

        # Summary logit bias (summary → MLP → vocab logit shift)
        self.summary_logit_bias = None
        if config.summary_logit_bias:
            vocab_size = model.config.vocab_size
            self.summary_logit_bias = SummaryLogitBias(
                d_summary=config.d_summary,
                vocab_size=vocab_size,
                hidden_dim=config.logit_bias_hidden_dim,
                device=self.device,
            )
            print(f"  Summary logit bias: {self.summary_logit_bias.param_count():,} params")

        # Freeze base model if configured (adapter-only training)
        if config.freeze_base_model:
            for p in model.parameters():
                p.requires_grad = False
            n_frozen = sum(p.numel() for p in model.parameters())
            print(f"  Froze base model: {n_frozen:,} params")

        # LoRA — lightweight adaptation of base model attention layers
        self.lora_applied = False
        if config.use_lora:
            from peft import LoraConfig, get_peft_model
            layers_to_transform = list(range(
                config.lora_layers_min, config.lora_layers_max + 1
            ))
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                layers_to_transform=layers_to_transform,
                bias="none",
            )
            self.model = get_peft_model(model, lora_config)
            self.lora_applied = True
            n_lora = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            print(f"  LoRA applied: {n_lora:,} trainable / {n_total:,} total "
                  f"(rank={config.lora_rank}, layers {config.lora_layers_min}-{config.lora_layers_max})")

        # Pseudo-token decoder — expand summary into diverse attention tokens
        self.pseudo_decoder = None
        if config.use_pseudo_tokens:
            self.pseudo_decoder = PseudoTokenDecoder(
                d_summary=config.d_summary,
                d_model=config.d_model,
                n_pseudo_tokens=config.n_pseudo_tokens,
                hidden_dim=config.pseudo_decoder_hidden,
                device=self.device,
            )
            print(f"  Pseudo-token decoder: {self.pseudo_decoder.param_count():,} params "
                  f"({config.n_pseudo_tokens} tokens)")

        # KV-cache prefix generator
        self.prefix_generator = None
        if config.use_kv_prefix:
            from src.model.prefix_generator import PrefixKVGenerator
            self.prefix_generator = PrefixKVGenerator(
                d_summary=config.d_summary,
                n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_attention_heads,
                d_head=config.d_model // model.config.num_attention_heads,
                hidden_dim=config.kv_prefix_hidden,
                active_layers_min=config.lora_layers_min,
            ).to(self.device)

        # Optimizer — separate param groups for different learning rates
        param_groups = []

        # Group 1: CCT components (commitment head, probe, up_project) — fast LR
        cct_params = []
        cct_params.extend(list(self.commitment_head.parameters()))
        cct_params.extend(list(self.sufficiency_probe.parameters()))
        cct_params.extend(list(self.summary_buffer.up_project.parameters()))
        if self.summary_attn_bias_param is not None:
            cct_params.append(self.summary_attn_bias_param)
        if self.summary_conditioner is not None:
            cct_params.extend(list(self.summary_conditioner.parameters()))
        if self.summary_adapter is not None:
            cct_params.extend(list(self.summary_adapter.parameters()))
        if self.summary_logit_bias is not None:
            cct_params.extend(list(self.summary_logit_bias.parameters()))
        if self.prefix_generator is not None:
            cct_params.extend(list(self.prefix_generator.parameters()))
        if self.pseudo_decoder is not None:
            cct_params.extend(list(self.pseudo_decoder.parameters()))
        param_groups.append({
            "params": cct_params,
            "lr": config.learning_rate,
            "name": "cct",
        })

        # Group 2: Base model params (full unfreeze or LoRA)
        if config.use_lora:
            # Only LoRA params are trainable — use separate LR
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            if lora_params:
                param_groups.append({
                    "params": lora_params,
                    "lr": config.lora_lr,
                    "name": "lora",
                })
        elif not config.freeze_base_model:
            base_params = [p for p in model.parameters() if p.requires_grad]
            if base_params:
                param_groups.append({
                    "params": base_params,
                    "lr": config.learning_rate,
                    "name": "base_model",
                })

        self.optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_steps,
        )

        self.global_step = 0

    def _make_bias_hook(self, layer_idx):
        """Create a forward pre-hook that adds per-head bias at summary positions.

        Each hook expands the (B, 1, S, S) causal mask to (B, H, S, S) and
        adds this layer's per-head bias at the first n_summary columns.
        Since -inf + bias = -inf, causal masking is preserved.
        """
        def hook(module, args, kwargs):
            n_sum = self._summary_n_tokens
            if n_sum <= 0:
                return None
            mask = kwargs.get('attention_mask')
            if mask is None:
                return None
            bias = self.summary_attn_bias_param[layer_idx]  # (n_heads,)
            n_heads = bias.shape[0]
            # Expand from (B, 1, S, S) to (B, H, S, S)
            if mask.size(1) == 1:
                mask = mask.expand(-1, n_heads, -1, -1).contiguous()
            else:
                mask = mask.clone()
            # Add per-head bias at summary columns
            bias_4d = bias.to(mask.dtype).view(1, n_heads, 1, 1)
            mask[:, :, :, :n_sum] = mask[:, :, :, :n_sum] + bias_4d
            kwargs['attention_mask'] = mask
            return args, kwargs
        return hook

    def _build_causal_mask(self, seq_len, batch_size, dtype):
        """Build standard 4D causal attention mask."""
        mask = torch.full(
            (seq_len, seq_len), float('-inf'), device=self.device, dtype=dtype
        )
        mask = torch.triu(mask, diagonal=1)  # -inf above diagonal, 0 on/below
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    def _get_amp_context(self):
        """Get automatic mixed precision context."""
        if self.config.mixed_precision == "bf16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        elif self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        elif self.device.type == "mps":
            # MPS: no autocast, use float32
            return torch.autocast("cpu", enabled=False)
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
        # Skip standard path when base model is frozen (no trainable params there)
        if not self.config.freeze_base_model:
            if phase == 1 or p_commit == 0.0:
                return self._standard_train_step(input_ids, weights)

            # Stochastic commitment: flip a coin per step boundary
            commit_this_step = (torch.rand(1).item() < p_commit)
            if not commit_this_step:
                return self._standard_train_step(input_ids, weights)

        # Update noise sigma from curriculum
        if self.config.noise_injection:
            self.commitment_head.noise_sigma = self.curriculum.get_noise_sigma(
                self.global_step
            )

        # Full CCT training step
        if self.config.training_mode == "sequential":
            return self._sequential_cct_train_step(input_ids, weights)
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
            prev_sum = None
            if self.config.recurrent_commitment and summaries:
                ps = summaries[-1]  # already detached
                if self.config.n_summary_tokens > 1:
                    ps = ps[:, 0, :]
                prev_sum = ps
            summary = self.commitment_head(step_hidden_f32, prev_summary=prev_sum)  # (B, d_summary)
            summaries_live.append(summary)  # keep grad for probe
            summary_detached = isolate_gradient(
                summary,
                mode=self.config.gradient_isolation,
                scale=self.config.gradient_scale,
            )
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

            # Inject up-projected summaries at STEP token positions
            if summaries:
                for idx, bp in enumerate(boundary_positions):
                    if idx < len(summaries):
                        up_proj = self.summary_buffer.up_project(
                            summaries[idx]
                        )  # (B, D)
                        if self.config.summary_injection_mode == "additive":
                            inputs_embeds[:, bp, :] = (
                                inputs_embeds[:, bp, :] + up_proj.to(inputs_embeds.dtype)
                            )
                        else:  # "replace"
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

    def _sequential_cct_train_step(
        self, input_ids: torch.Tensor, weights: dict
    ) -> dict:
        """Sequential single-pass CCT training step.

        Processes steps one at a time. Each step's input is:
          [prior_summary_0, ..., prior_summary_{k-1}, step_k_tokens]

        Cross-step token attention is blocked BY CONSTRUCTION — prior-step
        raw tokens are simply not in the input. Default causal mask suffices.

        With full_detach gradient isolation, each step's computation graph
        is independent, so we call loss.backward() per step to free memory.
        """
        batch_size, seq_len = input_ids.shape
        self.model.train()
        self.commitment_head.train()

        # Find STEP boundaries and step ranges
        boundary_positions = get_step_boundary_positions(
            input_ids[0].tolist(), self.step_token_id
        )
        step_ranges = get_step_ranges(seq_len, boundary_positions)

        # Get embedding layer
        if hasattr(self.model, "get_input_embeddings"):
            embed_layer = self.model.get_input_embeddings()
        else:
            embed_layer = self.model.gpt_neox.embed_in

        # Accumulators
        committed_summaries = []   # detached summaries for injection into next steps
        summaries_live = []        # live summaries for sufficiency probe gradient
        step_hidden_means = []     # probe reconstruction targets
        total_lm_loss = 0.0
        total_lm_tokens = 0

        # Multi-hop loss: live GRU chain with connected gradient
        live_chain = []            # non-detached GRU summaries for chain gradient
        step_base_logits_list = [] # pre-bias logits per step (detached) for L_hop
        step_labels_list = []      # token labels per step for L_hop

        can_backward_per_step = (self.config.gradient_isolation == "full_detach")
        accumulated_losses = []

        for step_idx, (start, end) in enumerate(step_ranges):
            is_last_step = (step_idx == len(step_ranges) - 1)
            step_len = end - start
            if step_len <= 0:
                continue

            with self._get_amp_context():
                # 1. Embed current step tokens
                step_token_ids = input_ids[:, start:end]  # (B, step_len)
                step_embeds = embed_layer(step_token_ids)  # (B, step_len, D)

                step_positions = torch.arange(
                    start, end, device=self.device
                ).unsqueeze(0).expand(batch_size, -1)  # (B, step_len)

                # 2. Prepare input: adapter (cross-attn) or conditioning (FiLM) or KV prefix or prepending (attention)
                K = self.config.n_summary_tokens
                n_prior_steps = len(committed_summaries)
                use_conditioning = (self.summary_conditioner is not None)
                use_adapter = (self.summary_adapter is not None)
                step_past_kv = None  # set by KV prefix path if active

                if use_adapter:
                    # Adapter path: no prepended tokens, set summary bank
                    n_summary_tokens = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    if n_prior_steps > 0:
                        # Stack all past summaries: (N, B, d_summary) -> (B, N, d_summary)
                        bank = torch.stack(committed_summaries, dim=0).transpose(0, 1)
                        if bank.dim() > 3:  # K > 1: flatten to (B, N*K, d_summary)
                            B_s, N_s, K_s, D_s = bank.shape
                            bank = bank.view(B_s, N_s * K_s, D_s)
                        self.summary_adapter.set_summary_bank(bank)
                elif use_conditioning:
                    # FiLM path: no summary tokens prepended
                    n_summary_tokens = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    # Set conditioning from aggregated prior summaries
                    if n_prior_steps > 0:
                        stacked = torch.stack(committed_summaries, dim=0)  # (N, B, ...)
                        agg = stacked.mean(dim=0)  # (B, d_summary) or (B, K, d_summary)
                        if agg.dim() > 2:  # K > 1: flatten to (B, d_summary)
                            agg = agg.mean(dim=1)
                        self.summary_conditioner.set_summary(agg)
                elif self.pseudo_decoder is not None and n_prior_steps > 0:
                    # Pseudo-token path: decode each summary into multiple tokens, prepend
                    P = self.config.n_pseudo_tokens
                    n_summary_tokens = n_prior_steps * P
                    summary_embeds_list = []
                    summary_pos_list = []
                    for s_idx in range(n_prior_steps):
                        s = committed_summaries[s_idx]
                        if K > 1:
                            s = s[:, 0, :]  # (B, d_summary) — use first token
                        decoded = self.pseudo_decoder(s).to(step_embeds.dtype)  # (B, P, D)
                        summary_embeds_list.append(decoded)
                        bp = boundary_positions[s_idx]
                        for p in range(P):
                            summary_pos_list.append(bp - P + 1 + p)

                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)
                    summary_positions = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=self.device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([summary_positions, step_positions], dim=1)
                elif self.pseudo_decoder is not None:
                    # First step, no summaries yet
                    n_summary_tokens = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                elif self.summary_logit_bias is not None:
                    # Logit bias path: no prepended tokens, bias applied post-forward
                    n_summary_tokens = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                elif self.prefix_generator is not None:
                    # KV prefix path: generate past_key_values, no prepended tokens
                    n_summary_tokens = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    if n_prior_steps > 0:
                        step_past_kv = self.prefix_generator(
                            committed_summaries[:n_prior_steps],
                            boundary_positions[:n_prior_steps],
                            batch_size,
                        )
                elif n_prior_steps > 0:
                    # Attention path: prepend summary embeddings
                    n_summary_tokens = n_prior_steps * K
                    summary_embeds_list = []
                    summary_pos_list = []
                    for s_idx in range(n_prior_steps):
                        s = committed_summaries[s_idx]
                        if K > 1:
                            up_proj = self.summary_buffer.up_project(s)  # (B, K, D)
                        else:
                            up_proj = self.summary_buffer.up_project(s).unsqueeze(1)  # (B, 1, D)
                        summary_embeds_list.append(up_proj.to(step_embeds.dtype))
                        bp = boundary_positions[s_idx]
                        for k in range(K):
                            summary_pos_list.append(bp - K + 1 + k)

                    summary_embeds = torch.cat(
                        summary_embeds_list, dim=1
                    )  # (B, n_summary_tokens, D)
                    inputs_embeds = torch.cat(
                        [summary_embeds, step_embeds], dim=1
                    )  # (B, n_summary_tokens + step_len, D)

                    summary_positions = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=self.device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat(
                        [summary_positions, step_positions], dim=1
                    )
                else:
                    n_summary_tokens = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                # 3. Forward pass
                attn_mask = None
                if self.summary_attn_bias_param is not None and not use_conditioning:
                    total_len = inputs_embeds.size(1)
                    attn_mask = self._build_causal_mask(
                        total_len, batch_size, inputs_embeds.dtype
                    )
                    self._summary_n_tokens = n_summary_tokens
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    past_key_values=step_past_kv,
                )
                self._summary_n_tokens = 0
                if use_conditioning:
                    self.summary_conditioner.clear()
                if use_adapter:
                    self.summary_adapter.clear()

                # 3.5 Apply logit bias from summary bank
                _step_bias = None  # saved for multi-hop pre-bias reconstruction
                if self.summary_logit_bias is not None and len(committed_summaries) > 0:
                    if self.config.last_summary_only:
                        lb_bank = committed_summaries[-1].unsqueeze(0).transpose(0, 1)
                    else:
                        lb_bank = torch.stack(committed_summaries, dim=0).transpose(0, 1)
                    if lb_bank.dim() > 3:
                        B_s, N_s, K_s, D_s = lb_bank.shape
                        lb_bank = lb_bank.view(B_s, N_s * K_s, D_s)
                    _step_bias = self.summary_logit_bias(lb_bank)  # (B, V)
                    outputs.logits = outputs.logits + _step_bias.unsqueeze(1)

                # 4. Commitment head (all steps except last)
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][
                        :, n_summary_tokens:, :
                    ]  # (B, step_len, D)
                    step_hidden_f32 = step_hidden.float()
                    step_hidden_means.append(
                        step_hidden_f32.mean(dim=1).detach()
                    )  # (B, D)

                    # For recurrent mode: pass previous summary (detached to prevent
                    # gradient flowing through the entire chain of steps)
                    prev_sum = None
                    if self.config.recurrent_commitment and committed_summaries:
                        ps = committed_summaries[-1]  # already detached
                        # If multi-token, use first token's summary for recurrence
                        if K > 1:
                            ps = ps[:, 0, :]  # (B, d_summary)
                        prev_sum = ps

                    if can_backward_per_step:
                        summary = self.commitment_head(
                            step_hidden_f32.detach(),
                            prev_summary=prev_sum,
                        )  # (B, d_summary) or (B, K, d_summary)
                    else:
                        summary = self.commitment_head(
                            step_hidden_f32,
                            prev_summary=prev_sum,
                        )  # (B, d_summary) or (B, K, d_summary)

                    # For sufficiency probe: flatten multi-token summary
                    if K > 1:
                        summaries_live.append(
                            summary.view(-1, K * self.config.d_summary)
                        )  # (B, K*d_summary)
                    else:
                        summaries_live.append(summary)

                    summary_detached = isolate_gradient(
                        summary,
                        mode=self.config.gradient_isolation,
                        scale=self.config.gradient_scale,
                    )
                    committed_summaries.append(summary_detached)

                    # Build live GRU chain for multi-hop loss (non-detached)
                    if self.config.multi_hop_loss and self.config.recurrent_commitment:
                        gru_inp = self.commitment_head._last_gru_input.detach()
                        prev_live = live_chain[-1] if live_chain else None
                        if prev_live is not None:
                            gru_h = self.commitment_head.gru(gru_inp, prev_live)
                        else:
                            gru_h = self.commitment_head.gru(gru_inp)
                        live_sum = self.commitment_head.gru_output(gru_h)
                        live_chain.append(live_sum)

                # 5. Per-step LM loss
                # Logits for step token positions predicting next step token
                step_logits = outputs.logits[
                    :, n_summary_tokens:n_summary_tokens + step_len - 1, :
                ]  # (B, step_len-1, V)

                step_labels = step_token_ids[:, 1:].clone()  # (B, step_len-1)
                step_labels[step_labels == self.step_token_id] = -100

                # Cross-boundary prediction: last summary token -> first step token
                if n_summary_tokens > 0:
                    cross_logit = outputs.logits[
                        :, n_summary_tokens - 1 : n_summary_tokens, :
                    ]  # (B, 1, V)
                    first_label = step_token_ids[:, 0:1].clone()  # (B, 1)
                    first_label[first_label == self.step_token_id] = -100

                    step_logits = torch.cat(
                        [cross_logit, step_logits], dim=1
                    )
                    step_labels = torch.cat(
                        [first_label, step_labels], dim=1
                    )

                # Save pre-bias step logits and labels for multi-hop loss
                if self.config.multi_hop_loss:
                    step_labels_list.append(step_labels.detach())
                    if _step_bias is not None:
                        # Subtract the standard bias to get pre-bias logits
                        pre_bias = (step_logits - _step_bias.unsqueeze(1)).detach()
                    else:
                        pre_bias = step_logits.detach()
                    step_base_logits_list.append(pre_bias)

                lm_loss = nn.functional.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    step_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (step_labels.reshape(-1) != -100).sum().item()
                total_lm_loss += lm_loss.item()
                total_lm_tokens += n_valid

                if n_valid > 0:
                    normalized_loss = (
                        weights["alpha"] * lm_loss / n_valid
                        / self.config.gradient_accumulation_steps
                    )
                else:
                    normalized_loss = lm_loss * 0.0

            # 6. Per-step backward (only with full_detach)
            if can_backward_per_step:
                if normalized_loss.requires_grad:
                    normalized_loss.backward()
            else:
                accumulated_losses.append(normalized_loss)

        # 7. Sufficiency probe loss (after all steps)
        # NOTE: L_suf and L_con both flow through summaries_live, so they must share
        # a single backward pass. We accumulate both into post_loop_losses.
        post_loop_losses = []
        suf_value = 0.0
        if weights["delta"] > 0 and summaries_live and step_hidden_means:
            with self._get_amp_context():
                summary_for_probe = torch.cat(summaries_live, dim=0)
                hidden_mean_for_probe = torch.cat(step_hidden_means, dim=0)
                reconstructed = self.sufficiency_probe(summary_for_probe)
                L_sufficiency = nn.functional.mse_loss(
                    reconstructed, hidden_mean_for_probe.detach()
                )
                suf_loss = (
                    weights["delta"] * L_sufficiency
                    / self.config.gradient_accumulation_steps
                )
                suf_value = L_sufficiency.item()
                post_loop_losses.append(suf_loss)

        # 7.5. L_conclusion_from_premises (after all steps)
        # Forces the model to learn to USE summaries for next-token prediction.
        # Uses LIVE (non-detached) summaries so gradient flows to commitment head,
        # training it to encode predictively useful information.
        # Gradient flows: loss → delivery mechanism → live summary → commitment head
        conclusion_value = 0.0
        use_conditioning = (self.summary_conditioner is not None)
        use_adapter = (self.summary_adapter is not None)
        if weights["gamma"] > 0 and summaries_live and self.config.conclusion_n_tokens > 0:
            conclusion_loss_sum = 0.0
            conclusion_token_count = 0

            for con_step_idx in range(1, len(step_ranges)):
                con_start, con_end = step_ranges[con_step_idx]
                n_target = min(self.config.conclusion_n_tokens, con_end - con_start)
                if n_target <= 1:
                    continue

                with self._get_amp_context():
                    K = self.config.n_summary_tokens

                    if use_adapter:
                        # Adapter path: set summary bank, forward target tokens
                        bank = torch.stack(
                            summaries_live[:con_step_idx], dim=0
                        ).transpose(0, 1)  # (B, N, d_summary)
                        if bank.dim() > 3:
                            B_s, N_s, K_s, D_s = bank.shape
                            bank = bank.view(B_s, N_s * K_s, D_s)
                        self.summary_adapter.set_summary_bank(bank)

                        con_target_ids = input_ids[:, con_start:con_start + n_target]
                        con_target_embeds = embed_layer(con_target_ids)
                        con_target_positions = torch.arange(
                            con_start, con_start + n_target, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)

                        con_outputs = self.model(
                            inputs_embeds=con_target_embeds,
                            position_ids=con_target_positions,
                        )
                        self.summary_adapter.clear()

                        con_logits = con_outputs.logits[:, :-1, :]
                        con_labels = con_target_ids[:, 1:].clone()
                        con_labels[con_labels == self.step_token_id] = -100
                    elif use_conditioning:
                        # FiLM path: condition model, forward just target tokens
                        stacked_con = torch.stack(
                            summaries_live[:con_step_idx], dim=0
                        ).mean(dim=0)
                        if stacked_con.dim() > 2:
                            stacked_con = stacked_con.mean(dim=1)
                        self.summary_conditioner.set_summary(stacked_con)

                        con_target_ids = input_ids[:, con_start:con_start + n_target]
                        con_target_embeds = embed_layer(con_target_ids)
                        con_target_positions = torch.arange(
                            con_start, con_start + n_target, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)

                        con_outputs = self.model(
                            inputs_embeds=con_target_embeds,
                            position_ids=con_target_positions,
                        )
                        self.summary_conditioner.clear()

                        # Predict: logits[:-1] → target_ids[1:]
                        con_logits = con_outputs.logits[:, :-1, :]
                        con_labels = con_target_ids[:, 1:].clone()
                        con_labels[con_labels == self.step_token_id] = -100
                    elif self.pseudo_decoder is not None:
                        # Pseudo-token path: decode live summaries, prepend + target
                        P = self.config.n_pseudo_tokens
                        con_summary_embeds_list = []
                        con_summary_pos_list = []
                        for j in range(con_step_idx):
                            s = summaries_live[j]
                            if K > 1:
                                s = s[:, 0, :]
                            decoded = self.pseudo_decoder(s).to(embed_layer.weight.dtype)
                            con_summary_embeds_list.append(decoded)
                            bp = boundary_positions[j]
                            for p in range(P):
                                con_summary_pos_list.append(bp - P + 1 + p)

                        con_summary_embeds = torch.cat(con_summary_embeds_list, dim=1)
                        n_sum = con_summary_embeds.size(1)

                        con_target_ids = input_ids[:, con_start:con_start + n_target]
                        con_target_embeds = embed_layer(con_target_ids)
                        con_target_positions = torch.arange(
                            con_start, con_start + n_target, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)

                        con_embeds = torch.cat([con_summary_embeds, con_target_embeds], dim=1)
                        con_summary_pos = torch.tensor(
                            con_summary_pos_list, dtype=torch.long, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)
                        con_positions = torch.cat([con_summary_pos, con_target_positions], dim=1)

                        con_outputs = self.model(
                            inputs_embeds=con_embeds,
                            position_ids=con_positions,
                        )

                        con_logits = con_outputs.logits[:, n_sum - 1 : n_sum + n_target - 1, :]
                        con_labels = con_target_ids.clone()
                        con_labels[con_labels == self.step_token_id] = -100
                    elif self.summary_logit_bias is not None:
                        # Logit bias path: forward only target tokens, bias applied below
                        con_target_ids = input_ids[:, con_start:con_start + n_target]
                        con_target_embeds = embed_layer(con_target_ids)
                        con_target_positions = torch.arange(
                            con_start, con_start + n_target, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)

                        con_outputs = self.model(
                            inputs_embeds=con_target_embeds,
                            position_ids=con_target_positions,
                        )

                        con_logits = con_outputs.logits[:, :-1, :]
                        con_labels = con_target_ids[:, 1:].clone()
                        con_labels[con_labels == self.step_token_id] = -100
                    elif self.prefix_generator is not None:
                        # KV prefix path: generate past_key_values from live summaries
                        con_past_kv = self.prefix_generator(
                            summaries_live[:con_step_idx],
                            boundary_positions[:con_step_idx],
                            batch_size,
                        )
                        # Include bridge token (last of prev step) so logits[0]
                        # predicts the first target token — the most summary-dependent.
                        con_full_ids = input_ids[:, con_start - 1:con_start + n_target]
                        con_full_embeds = embed_layer(con_full_ids)
                        con_full_positions = torch.arange(
                            con_start - 1, con_start + n_target, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)

                        con_outputs = self.model(
                            inputs_embeds=con_full_embeds,
                            position_ids=con_full_positions,
                            past_key_values=con_past_kv,
                        )

                        con_logits = con_outputs.logits[:, :-1, :]
                        con_labels = con_full_ids[:, 1:].clone()
                        con_labels[con_labels == self.step_token_id] = -100
                    else:
                        # Attention path: prepend summary tokens + target tokens
                        con_summary_embeds_list = []
                        con_summary_pos_list = []
                        for j in range(con_step_idx):
                            s = summaries_live[j]
                            if K > 1:
                                up = self.summary_buffer.up_project(s)
                            else:
                                up = self.summary_buffer.up_project(s).unsqueeze(1)
                            con_summary_embeds_list.append(up.to(embed_layer.weight.dtype))
                            bp = boundary_positions[j]
                            for k in range(K):
                                con_summary_pos_list.append(bp - K + 1 + k)

                        con_summary_embeds = torch.cat(con_summary_embeds_list, dim=1)
                        n_sum = con_summary_embeds.size(1)

                        con_target_ids = input_ids[:, con_start:con_start + n_target]
                        con_target_embeds = embed_layer(con_target_ids)
                        con_target_positions = torch.arange(
                            con_start, con_start + n_target, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)

                        con_embeds = torch.cat([con_summary_embeds, con_target_embeds], dim=1)
                        con_summary_pos = torch.tensor(
                            con_summary_pos_list, dtype=torch.long, device=self.device
                        ).unsqueeze(0).expand(batch_size, -1)
                        con_positions = torch.cat([con_summary_pos, con_target_positions], dim=1)

                        con_attn_mask = None
                        if self.summary_attn_bias_param is not None:
                            con_total_len = con_embeds.size(1)
                            con_attn_mask = self._build_causal_mask(
                                con_total_len, batch_size, con_embeds.dtype
                            )
                            self._summary_n_tokens = n_sum
                        con_outputs = self.model(
                            inputs_embeds=con_embeds,
                            position_ids=con_positions,
                            attention_mask=con_attn_mask,
                        )
                        self._summary_n_tokens = 0

                        con_logits = con_outputs.logits[:, n_sum - 1 : n_sum + n_target - 1, :]
                        con_labels = con_target_ids.clone()
                        con_labels[con_labels == self.step_token_id] = -100

                    # Apply logit bias to conclusion logits
                    if self.summary_logit_bias is not None:
                        if self.config.last_summary_only:
                            con_lb_bank = summaries_live[con_step_idx - 1].unsqueeze(0).transpose(0, 1)
                        else:
                            con_lb_bank = torch.stack(
                                summaries_live[:con_step_idx], dim=0
                            ).transpose(0, 1)
                        if con_lb_bank.dim() > 3:
                            B_s, N_s, K_s, D_s = con_lb_bank.shape
                            con_lb_bank = con_lb_bank.view(B_s, N_s * K_s, D_s)
                        con_bias = self.summary_logit_bias(con_lb_bank)
                        con_logits = con_logits + con_bias.unsqueeze(1)

                    con_loss = nn.functional.cross_entropy(
                        con_logits.reshape(-1, con_logits.size(-1)),
                        con_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    n_valid = (con_labels.reshape(-1) != -100).sum().item()

                    if n_valid > 0:
                        conclusion_loss_sum += con_loss.item()
                        conclusion_token_count += n_valid

                        normalized_con = (
                            weights["gamma"] * con_loss / n_valid
                            / self.config.gradient_accumulation_steps
                        )

                        # Accumulate into post_loop_losses — L_con shares the
                        # summaries_live graph with L_suf, so they must backward together.
                        post_loop_losses.append(normalized_con)

            if conclusion_token_count > 0:
                conclusion_value = conclusion_loss_sum / conclusion_token_count

        # 7.7. Multi-hop loss: gradient through full GRU chain
        # Uses the live (non-detached) GRU chain to create gradient pressure
        # for information to survive across multiple recurrent steps.
        # For each step K, applies live_chain[K]'s logit bias to step K's
        # pre-bias logits. Gradient flows: L_hop → logit_bias → live_chain[K]
        # → GRU → live_chain[K-1] → ... → live_chain[0].
        hop_value = 0.0
        if (self.config.multi_hop_loss and self.config.recurrent_commitment
                and self.summary_logit_bias is not None
                and len(live_chain) >= 2 and len(step_base_logits_list) >= 2):
            hop_loss_sum = 0.0
            hop_token_count = 0
            n_summary_tokens = self.config.n_summary_tokens

            # Progressive curriculum: which hop distances are active
            progress = self.global_step / max(self.config.total_steps, 1)
            if progress < 0.25:
                max_hop = 3
            elif progress < 0.50:
                max_hop = 5
            else:
                max_hop = 8

            # Hop weights: 2→0.3, 3-4→0.4, 5-8→0.3
            def hop_weight(d):
                if d == 2:
                    return 0.3
                elif d <= 4:
                    return 0.2  # 0.4 split across 3,4
                else:
                    return 0.075  # 0.3 split across 5,6,7,8

            with self._get_amp_context():
                for target_step in range(1, len(live_chain)):
                    for d in range(2, min(max_hop + 1, target_step + 1)):
                        source_step = target_step - d
                        if source_step < 0:
                            continue

                        # live_chain[target_step] carries info from source through chain
                        live_sum = live_chain[target_step]  # (B, d_summary*K)
                        lb_bank = live_sum.unsqueeze(1)  # (B, 1, d_summary*K)
                        hop_bias = self.summary_logit_bias(lb_bank)  # (B, V)

                        # Pre-bias logits already sliced to match step_labels
                        base_logits = step_base_logits_list[target_step]
                        target_labels = step_labels_list[target_step]
                        hop_logits = base_logits + hop_bias.unsqueeze(1)

                        hop_ce = nn.functional.cross_entropy(
                            hop_logits.reshape(-1, hop_logits.size(-1)),
                            target_labels.reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        n_valid = (target_labels.reshape(-1) != -100).sum().item()

                        if n_valid > 0:
                            w = hop_weight(d)
                            hop_loss_sum += hop_ce.item() * w
                            hop_token_count += n_valid

                            normalized_hop = (
                                self.config.multi_hop_weight * w * hop_ce / n_valid
                                / self.config.gradient_accumulation_steps
                            )
                            post_loop_losses.append(normalized_hop)

            if hop_token_count > 0:
                hop_value = hop_loss_sum / hop_token_count

        # 7.6. Single backward for all post-loop losses (L_suf + L_con + L_hop)
        # All share live computation graphs, so must backward together.
        if post_loop_losses:
            total_post = sum(post_loop_losses)
            total_post.backward()
            del post_loop_losses

        # 8. Accumulated backward (for scaled/none isolation)
        if not can_backward_per_step and accumulated_losses:
            total_accumulated = sum(accumulated_losses)
            total_accumulated.backward()

        # 9. Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            all_params = list(self.commitment_head.parameters()) + \
                list(self.sufficiency_probe.parameters()) + \
                list(self.summary_buffer.up_project.parameters())
            if self.config.use_lora:
                all_params.extend([p for p in self.model.parameters() if p.requires_grad])
            elif not self.config.freeze_base_model:
                all_params.extend(list(self.model.parameters()))
            if self.summary_attn_bias_param is not None:
                all_params.append(self.summary_attn_bias_param)
            if self.summary_conditioner is not None:
                all_params.extend(list(self.summary_conditioner.parameters()))
            if self.summary_adapter is not None:
                all_params.extend(list(self.summary_adapter.parameters()))
            if self.summary_logit_bias is not None:
                all_params.extend(list(self.summary_logit_bias.parameters()))
            if self.pseudo_decoder is not None:
                all_params.extend(list(self.pseudo_decoder.parameters()))
            nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        avg_lm_loss = total_lm_loss / total_lm_tokens if total_lm_tokens > 0 else 0.0
        loss_dict = {
            "total": (weights["alpha"] * avg_lm_loss
                      + weights.get("delta", 0) * suf_value
                      + weights.get("gamma", 0) * conclusion_value),
            "standard": avg_lm_loss,
            "validity": avg_lm_loss,  # same as standard in sequential mode
            "conclusion": conclusion_value,
            "sufficiency": suf_value,
            "phase": self.curriculum.get_phase(self.global_step),
            "p_commit": self.curriculum.get_commitment_probability(self.global_step),
            "lr": self.scheduler.get_last_lr()[0],
            "n_summaries": len(committed_summaries) * self.config.n_summary_tokens,
        }
        if hasattr(self.commitment_head, '_last_gate_mean'):
            loss_dict["gate_mean"] = self.commitment_head._last_gate_mean
        if self.config.multi_hop_loss:
            loss_dict["hop"] = hop_value
        if self.summary_adapter is not None:
            gate_vals = torch.sigmoid(self.summary_adapter.gates).detach()
            loss_dict["adapter_gate_mean"] = gate_vals.mean().item()
            loss_dict["adapter_gate_max"] = gate_vals.max().item()
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
                "summary_attn_bias": self.summary_attn_bias_param.data if self.summary_attn_bias_param is not None else None,
                "summary_conditioner_state_dict": self.summary_conditioner.state_dict() if self.summary_conditioner is not None else None,
                "summary_adapter_state_dict": self.summary_adapter.state_dict() if self.summary_adapter is not None else None,
                "summary_logit_bias_state_dict": self.summary_logit_bias.state_dict() if self.summary_logit_bias is not None else None,
                "prefix_generator_state_dict": self.prefix_generator.state_dict() if self.prefix_generator is not None else None,
                "pseudo_decoder_state_dict": self.pseudo_decoder.state_dict() if self.pseudo_decoder is not None else None,
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
        if "summary_attn_bias" in ckpt and ckpt["summary_attn_bias"] is not None and self.summary_attn_bias_param is not None:
            self.summary_attn_bias_param.data = ckpt["summary_attn_bias"].to(self.device)
        if "summary_conditioner_state_dict" in ckpt and ckpt["summary_conditioner_state_dict"] is not None and self.summary_conditioner is not None:
            self.summary_conditioner.load_state_dict(ckpt["summary_conditioner_state_dict"])
        if "summary_adapter_state_dict" in ckpt and ckpt["summary_adapter_state_dict"] is not None and self.summary_adapter is not None:
            self.summary_adapter.load_state_dict(ckpt["summary_adapter_state_dict"])
        if "summary_logit_bias_state_dict" in ckpt and ckpt["summary_logit_bias_state_dict"] is not None and self.summary_logit_bias is not None:
            self.summary_logit_bias.load_state_dict(ckpt["summary_logit_bias_state_dict"])
        if "prefix_generator_state_dict" in ckpt and ckpt["prefix_generator_state_dict"] is not None and self.prefix_generator is not None:
            self.prefix_generator.load_state_dict(ckpt["prefix_generator_state_dict"])
        if "pseudo_decoder_state_dict" in ckpt and ckpt["pseudo_decoder_state_dict"] is not None and self.pseudo_decoder is not None:
            self.pseudo_decoder.load_state_dict(ckpt["pseudo_decoder_state_dict"])
