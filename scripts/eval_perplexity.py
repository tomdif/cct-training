"""
Perplexity comparison: CCT model vs baseline on the same held-out data.

Evaluates THREE conditions:
  1. Baseline model (standard attention)
  2. CCT model with standard attention (weights-only test)
  3. CCT model with CCT attention mask + summary injection (deployment scenario)

Usage:
  python scripts/eval_perplexity.py --config configs/tier2_410m.yaml \
    --cct-checkpoint ./checkpoints-tier2/cct-final.pt \
    --baseline-dir ./checkpoints-tier2/baseline-final \
    --eval-batches 200
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_pipeline import (
    extend_tokenizer,
    annotate_step_boundaries,
    get_step_boundary_positions,
    get_step_ranges,
)
from src.model.commitment_head import CommitmentHead
from src.model.summary_buffer import SummaryBuffer
from src.model.summary_conditioner import SummaryConditioner
from src.model.summary_adapter import SummaryAdapter
from src.model.summary_logit_bias import SummaryLogitBias
from src.model.cct_attention import build_cct_attention_mask_fast
from src.model.model_utils import get_model_layers


class StreamingEvalDataset(torch.utils.data.IterableDataset):
    """Deterministic eval data — skips first `skip_examples` to avoid train overlap."""

    def __init__(self, dataset_name, split, tokenizer, seq_len, step_token_id,
                 step_length, skip_examples=50000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.step_token_id = step_token_id
        self.step_length = step_length
        self.skip_examples = skip_examples

        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def __iter__(self):
        buffer = []
        skipped = 0
        for example in self.dataset:
            if skipped < self.skip_examples:
                skipped += 1
                continue
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                annotated = annotate_step_boundaries(
                    chunk, self.step_token_id, "fixed", self.step_length
                )
                annotated = annotated[:self.seq_len]
                yield torch.tensor(annotated, dtype=torch.long)


def compute_perplexity(model, dataloader, step_token_id, device, max_batches=200):
    """Compute perplexity with standard causal attention. Masks STEP tokens from loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            labels = input_ids.clone()
            labels[labels == step_token_id] = -100
            labels = labels[:, 1:]

            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (labels.reshape(-1) != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid
            n_batches += 1

            if n_batches % 50 == 0:
                avg = total_loss / total_tokens
                print(f"  batch {n_batches}/{max_batches} | avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens, n_batches


def compute_perplexity_cct_masked(
    model, commitment_head, up_project, dataloader,
    step_token_id, device, max_batches=200,
    injection_mode="replace",
):
    """Compute perplexity with CCT attention mask and summary injection.

    This replicates the actual CCT deployment scenario:
    1. Pass 1: standard forward to get hidden states
    2. Compute commitment summaries at step boundaries
    3. Pass 2: forward with CCT mask + injected summaries
    4. Measure loss on Pass 2 logits
    """
    model.eval()
    commitment_head.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            # Find step boundaries (use first batch element — uniform annotation)
            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(seq_len, boundary_positions)

            # === Pass 1: standard forward to get hidden states ===
            outputs_pass1 = model(input_ids, output_hidden_states=True)
            hidden_states = outputs_pass1.hidden_states[-1]  # (B, S, D)

            # Compute commitment summaries at each step boundary
            summaries = []
            for start, end in step_ranges[:-1]:  # all steps except last
                step_hidden = hidden_states[:, start:end, :]  # (B, step_len, D)
                step_hidden_f32 = step_hidden.float()
                prev_sum = summaries[-1] if summaries else None
                summary = commitment_head(step_hidden_f32, prev_summary=prev_sum)  # (B, d_summary)
                summaries.append(summary)

            # === Pass 2: CCT-masked forward ===
            if hasattr(model, "get_input_embeddings"):
                embed_layer = model.get_input_embeddings()
            else:
                embed_layer = model.gpt_neox.embed_in

            inputs_embeds = embed_layer(input_ids)  # (B, S, D)

            # Inject up-projected summaries at STEP token positions
            if summaries:
                for idx, bp in enumerate(boundary_positions):
                    if idx < len(summaries):
                        up_proj = up_project(summaries[idx])  # (B, D)
                        if injection_mode == "additive":
                            inputs_embeds[:, bp, :] = (
                                inputs_embeds[:, bp, :] + up_proj.to(inputs_embeds.dtype)
                            )
                        else:  # "replace"
                            inputs_embeds[:, bp, :] = up_proj.to(inputs_embeds.dtype)

            # Build CCT attention mask
            cct_mask = build_cct_attention_mask_fast(
                seq_len=seq_len,
                boundary_positions=boundary_positions,
                num_prior_summaries=0,
                device=device,
                batch_size=batch_size,
            )

            # Forward with CCT mask
            outputs_pass2 = model(
                inputs_embeds=inputs_embeds,
                attention_mask=cct_mask,
            )
            logits = outputs_pass2.logits[:, :-1, :]

            # Labels: mask STEP tokens
            labels = input_ids.clone()
            labels[labels == step_token_id] = -100
            labels = labels[:, 1:]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (labels.reshape(-1) != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid
            n_batches += 1

            if n_batches % 50 == 0:
                avg = total_loss / total_tokens
                print(f"  batch {n_batches}/{max_batches} | avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens, n_batches


def _build_causal_mask_with_bias(total_len, n_summary, batch_size, bias_param, device, dtype):
    """Build causal mask with per-head-per-layer bias on summary columns.

    Returns a (B, 1, S, S) base mask. Per-layer bias is applied by hooks.
    If no bias, returns None (use default causal mask).
    """
    if bias_param is None or n_summary <= 0:
        return None
    mask = torch.full((total_len, total_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)


class _SummaryBiasHooks:
    """Manages per-layer attention bias hooks for eval."""
    def __init__(self, model, bias_param):
        self.bias_param = bias_param
        self.n_summary = 0
        self.hooks = []
        if bias_param is not None:
            n_layers = bias_param.shape[0]
            for i in range(n_layers):
                hook = get_model_layers(model)[i].register_forward_pre_hook(
                    self._make_hook(i), with_kwargs=True,
                )
                self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs):
            n_sum = self.n_summary
            if n_sum <= 0:
                return None
            mask = kwargs.get('attention_mask')
            if mask is None:
                return None
            bias = self.bias_param[layer_idx]
            n_heads = bias.shape[0]
            if mask.size(1) == 1:
                mask = mask.expand(-1, n_heads, -1, -1).contiguous()
            else:
                mask = mask.clone()
            bias_4d = bias.to(mask.dtype).view(1, n_heads, 1, 1)
            mask[:, :, :, :n_sum] = mask[:, :, :, :n_sum] + bias_4d
            kwargs['attention_mask'] = mask
            return args, kwargs
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def compute_perplexity_sequential(
    model, commitment_head, up_project, dataloader,
    step_token_id, device, max_batches=200,
    n_summary_tokens=1,
    summary_attn_bias=None,
    summary_conditioner=None,
    summary_adapter=None,
    summary_logit_bias=None,
    prefix_generator=None,
):
    """Compute perplexity with sequential single-pass evaluation.

    Matches the sequential training procedure exactly:
    - Process each step with [prior_summaries, current_tokens]
    - Commitment head sees CCT-masked hidden states (same as deployment)
    - Supports multi-token summaries (K > 1).
    """
    model.eval()
    commitment_head.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    K = n_summary_tokens

    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.gpt_neox.embed_in

    # Register per-layer attention bias hooks if bias param provided
    bias_ctx = _SummaryBiasHooks(model, summary_attn_bias)

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch.to(device)
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            boundary_positions = get_step_boundary_positions(
                input_ids[0].tolist(), step_token_id
            )
            step_ranges = get_step_ranges(seq_len, boundary_positions)

            committed_summaries = []

            for step_idx, (start, end) in enumerate(step_ranges):
                step_len = end - start
                if step_len <= 0:
                    continue
                step_token_ids = input_ids[:, start:end]
                step_embeds = embed_layer(step_token_ids)

                step_positions = torch.arange(
                    start, end, device=device
                ).unsqueeze(0).expand(batch_size, -1)

                n_prior_steps = len(committed_summaries)
                step_past_kv = None  # set by KV prefix path if active

                if summary_adapter is not None:
                    # Adapter path: no prepended tokens, set summary bank
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    if n_prior_steps > 0:
                        bank = torch.stack(committed_summaries, dim=0).transpose(0, 1)
                        if bank.dim() > 3:
                            B_s, N_s, K_s, D_s = bank.shape
                            bank = bank.view(B_s, N_s * K_s, D_s)
                        summary_adapter.set_summary_bank(bank)
                elif summary_conditioner is not None:
                    # FiLM conditioning path: no prepended tokens
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    if n_prior_steps > 0:
                        stacked = torch.stack(committed_summaries, dim=0).mean(dim=0)
                        if stacked.dim() > 2:
                            stacked = stacked.mean(dim=1)
                        summary_conditioner.set_summary(stacked)
                elif summary_logit_bias is not None:
                    # Logit bias path: no prepended tokens, bias applied post-forward
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                elif prefix_generator is not None:
                    # KV prefix path: generate past_key_values, no prepended tokens
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions
                    if n_prior_steps > 0:
                        step_past_kv = prefix_generator(
                            committed_summaries[:n_prior_steps],
                            boundary_positions[:n_prior_steps],
                            batch_size,
                        )
                        # Cast KV cache to model dtype (prefix gen runs in float32)
                        for layer in step_past_kv.layers:
                            layer.keys = layer.keys.to(step_embeds.dtype)
                            layer.values = layer.values.to(step_embeds.dtype)
                elif n_prior_steps > 0:
                    # Attention path: prepend summary tokens
                    n_prepend = n_prior_steps * K
                    summary_embeds_list = []
                    summary_pos_list = []
                    for s_idx in range(n_prior_steps):
                        s = committed_summaries[s_idx]
                        if K > 1:
                            up_proj = up_project(s).to(step_embeds.dtype)  # (B, K, D)
                        else:
                            up_proj = up_project(s).unsqueeze(1).to(step_embeds.dtype)  # (B, 1, D)
                        summary_embeds_list.append(up_proj)
                        bp = boundary_positions[s_idx]
                        for k in range(K):
                            summary_pos_list.append(bp - K + 1 + k)

                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    inputs_embeds = torch.cat(
                        [summary_embeds, step_embeds], dim=1
                    )
                    summary_positions = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat(
                        [summary_positions, step_positions], dim=1
                    )
                else:
                    n_prepend = 0
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                # Build attention mask with summary bias if configured
                attn_mask = None
                if summary_attn_bias is not None and n_prepend > 0:
                    total_len = inputs_embeds.size(1)
                    attn_mask = _build_causal_mask_with_bias(
                        total_len, n_prepend, batch_size,
                        summary_attn_bias, device, inputs_embeds.dtype,
                    )
                    bias_ctx.n_summary = n_prepend
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    past_key_values=step_past_kv,
                )
                bias_ctx.n_summary = 0
                if summary_conditioner is not None:
                    summary_conditioner.clear()
                if summary_adapter is not None:
                    summary_adapter.clear()

                # Apply logit bias from summary bank
                if summary_logit_bias is not None and len(committed_summaries) > 0:
                    if config.get("last_summary_only", False):
                        lb_bank = committed_summaries[-1].unsqueeze(0).transpose(0, 1)
                    else:
                        lb_bank = torch.stack(committed_summaries, dim=0).transpose(0, 1)
                    if lb_bank.dim() > 3:
                        B_s, N_s, K_s, D_s = lb_bank.shape
                        lb_bank = lb_bank.view(B_s, N_s * K_s, D_s)
                    bias = summary_logit_bias(lb_bank)  # (B, V)
                    outputs.logits = outputs.logits + bias.unsqueeze(1)

                # Commitment summary for non-last steps
                is_last_step = (step_idx == len(step_ranges) - 1)
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][:, n_prepend:, :]
                    prev_sum = committed_summaries[-1] if committed_summaries else None
                    summary = commitment_head(step_hidden.float(), prev_summary=prev_sum)
                    committed_summaries.append(summary)

                # Loss on step tokens
                step_logits = outputs.logits[
                    :, n_prepend:n_prepend + step_len - 1, :
                ]
                step_labels = step_token_ids[:, 1:].clone()
                step_labels[step_labels == step_token_id] = -100

                # Cross-boundary prediction
                if n_prepend > 0:
                    cross_logit = outputs.logits[
                        :, n_prepend - 1:n_prepend, :
                    ]
                    first_label = step_token_ids[:, 0:1].clone()
                    first_label[first_label == step_token_id] = -100
                    step_logits = torch.cat(
                        [cross_logit, step_logits], dim=1
                    )
                    step_labels = torch.cat(
                        [first_label, step_labels], dim=1
                    )

                loss = torch.nn.functional.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    step_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (step_labels.reshape(-1) != -100).sum().item()
                total_loss += loss.item()
                total_tokens += n_valid

            n_batches += 1
            if n_batches % 50 == 0:
                avg = total_loss / total_tokens
                print(f"  batch {n_batches}/{max_batches} | avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    # Cleanup hooks
    bias_ctx.remove()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens, n_batches


def main():
    parser = argparse.ArgumentParser(description="CCT vs Baseline Perplexity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--eval-batches", type=int, default=200)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) backend")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found, evaluation on CPU (slow)")
    model_name = config["base_model"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_token_id = extend_tokenizer(tokenizer)

    # Build eval dataset (same for all conditions)
    eval_dataset = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )

    batch_size = config.get("batch_size", 4)

    # ══════════════════════════════════════════════════════════
    # CONDITION 1: BASELINE (standard attention)
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("CONDITION 1: BASELINE (standard attention)")
    print("=" * 60)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_dir,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    bl_loss, bl_ppl, bl_tokens, bl_batches = compute_perplexity(
        baseline_model, eval_loader, step_token_id, device, args.eval_batches
    )
    print(f"\n  Baseline: loss={bl_loss:.4f}  ppl={bl_ppl:.2f}  ({bl_tokens} tokens, {bl_batches} batches)")

    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # CONDITION 2: CCT (standard attention — weights-only test)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 2: CCT (standard attention — weights-only)")
    print("=" * 60)
    cct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    cct_model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)

    # Apply LoRA wrapping if checkpoint was trained with LoRA
    if config.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        from src.model.model_utils import detect_lora_target_modules
        target_modules = config.get("lora_target_modules", [])
        if not target_modules:
            target_modules = detect_lora_target_modules(cct_model)
            print(f"  LoRA auto-detected target modules: {target_modules}")
        lora_config = LoraConfig(
            r=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=0.0,
            target_modules=target_modules,
            layers_to_transform=list(range(
                config.get("lora_layers_min", 12),
                config.get("lora_layers_max", 23) + 1,
            )),
            bias="none",
        )
        cct_model = get_peft_model(cct_model, lora_config)

    # Cast checkpoint weights to model dtype (LoRA weights may be float32 from AMP training)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cct_state = {k: v.to(model_dtype) if v.is_floating_point() else v for k, v in ckpt["model_state_dict"].items()}
    cct_model.load_state_dict(cct_state)

    eval_loader2 = DataLoader(eval_dataset, batch_size=batch_size)
    cct_std_loss, cct_std_ppl, cct_std_tokens, cct_std_batches = compute_perplexity(
        cct_model, eval_loader2, step_token_id, device, args.eval_batches
    )
    print(f"\n  CCT (std attn): loss={cct_std_loss:.4f}  ppl={cct_std_ppl:.2f}  ({cct_std_tokens} tokens, {cct_std_batches} batches)")

    # ══════════════════════════════════════════════════════════
    # CONDITION 3: CCT (CCT attention mask — deployment scenario)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 3: CCT (CCT mask + summaries — deployment)")
    print("=" * 60)

    # Load commitment head and up-projection (match training config)
    commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config["d_summary"],
        d_bottleneck=config["d_bottleneck"],
        use_tanh=config.get("use_tanh", True),
        use_l2_norm=config.get("use_l2_norm", True),
        noise_injection=config.get("noise_injection", False),
        n_summary_tokens=config.get("n_summary_tokens", 1),
        recurrent=config.get("recurrent_commitment", False),
    ).to(device)
    commitment_head.load_state_dict(ckpt["commitment_head_state_dict"])

    summary_buffer = SummaryBuffer(
        d_summary=config["d_summary"],
        d_model=config["d_model"],
        device=device,
        decoder_type=config.get("decoder_type", "linear"),
        decoder_bottleneck=config.get("decoder_bottleneck", None),
    )
    summary_buffer.up_project.load_state_dict(ckpt["summary_up_project_state_dict"])
    up_project = summary_buffer.up_project

    # Load summary attention bias if present
    summary_attn_bias = ckpt.get("summary_attn_bias", None)
    if summary_attn_bias is not None:
        summary_attn_bias = summary_attn_bias.to(device)
        print(f"  Loaded summary attention bias: shape={summary_attn_bias.shape}, mean={summary_attn_bias.mean():.2f}")

    # Load summary conditioner if present
    summary_conditioner = None
    if ckpt.get("summary_conditioner_state_dict") is not None:
        n_layers = cct_model.config.num_hidden_layers
        summary_conditioner = SummaryConditioner(
            d_summary=config["d_summary"],
            d_model=config["d_model"],
            n_layers=n_layers,
            device=device,
        )
        summary_conditioner.load_state_dict(ckpt["summary_conditioner_state_dict"])
        summary_conditioner.register_hooks(cct_model)
        n_cond_params = sum(p.numel() for p in summary_conditioner.parameters())
        print(f"  Loaded summary conditioner: {n_cond_params:,} params")

    n_summary_tokens = config.get("n_summary_tokens", 1)
    if n_summary_tokens > 1:
        # 2-pass deployment doesn't support multi-token summaries (single STEP position)
        print("  SKIPPED: 2-pass deployment not compatible with n_summary_tokens > 1")
        cct_mask_loss, cct_mask_ppl = 0.0, 0.0
        cct_mask_tokens, cct_mask_batches = 0, 0
    else:
        eval_loader3 = DataLoader(eval_dataset, batch_size=batch_size)
        injection_mode = config.get("summary_injection", "replace")
        cct_mask_loss, cct_mask_ppl, cct_mask_tokens, cct_mask_batches = compute_perplexity_cct_masked(
            cct_model, commitment_head, up_project, eval_loader3,
            step_token_id, device, args.eval_batches,
            injection_mode=injection_mode,
        )
        print(f"\n  CCT (masked): loss={cct_mask_loss:.4f}  ppl={cct_mask_ppl:.2f}  ({cct_mask_tokens} tokens, {cct_mask_batches} batches)")

    # Load summary adapter if present
    summary_adapter = None
    if ckpt.get("summary_adapter_state_dict") is not None:
        n_layers = cct_model.config.num_hidden_layers
        adapter_d = config.get("adapter_d", 64)
        summary_adapter = SummaryAdapter(
            d_model=config["d_model"],
            d_summary=config["d_summary"],
            n_layers=n_layers,
            d_adapter=adapter_d,
            device=device,
        )
        summary_adapter.load_state_dict(ckpt["summary_adapter_state_dict"])
        summary_adapter.register_hooks(cct_model)
        print(f"  Loaded summary adapter: {summary_adapter.param_count():,} params")

    # Load summary logit bias if present
    summary_logit_bias = None
    if ckpt.get("summary_logit_bias_state_dict") is not None:
        summary_logit_bias = SummaryLogitBias(
            d_summary=config["d_summary"],
            vocab_size=cct_model.config.vocab_size,
            hidden_dim=config.get("logit_bias_hidden_dim", 256),
            device=device,
        )
        summary_logit_bias.load_state_dict(ckpt["summary_logit_bias_state_dict"])
        print(f"  Loaded summary logit bias: {summary_logit_bias.param_count():,} params")

    # Load prefix generator if present
    prefix_generator = None
    if ckpt.get("prefix_generator_state_dict") is not None:
        from src.model.prefix_generator import PrefixKVGenerator
        prefix_generator = PrefixKVGenerator(
            d_summary=config["d_summary"],
            n_layers=cct_model.config.num_hidden_layers,
            n_heads=cct_model.config.num_attention_heads,
            d_head=config["d_model"] // cct_model.config.num_attention_heads,
            hidden_dim=config.get("kv_prefix_hidden", 512),
            active_layers_min=config.get("lora_layers_min", 12),
        ).to(device)
        prefix_generator.load_state_dict(ckpt["prefix_generator_state_dict"])
        prefix_generator.eval()
        print(f"  Loaded prefix generator: {prefix_generator.param_count():,} params")

    # ══════════════════════════════════════════════════════════
    # CONDITION 4: CCT (Sequential — matched train/deploy)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CONDITION 4: CCT (Sequential -- matched train/deploy)")
    print("=" * 60)
    eval_dataset4 = StreamingEvalDataset(
        dataset_name=config.get("validation_dataset", config["dataset"]),
        split=config.get("validation_split", "train"),
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        step_token_id=step_token_id,
        step_length=config["step_length"],
        skip_examples=50000,
    )
    eval_loader4 = DataLoader(eval_dataset4, batch_size=batch_size)
    n_summary_tokens = config.get("n_summary_tokens", 1)
    seq_loss, seq_ppl, seq_tokens, seq_batches = compute_perplexity_sequential(
        cct_model, commitment_head, up_project, eval_loader4,
        step_token_id, device, args.eval_batches,
        n_summary_tokens=n_summary_tokens,
        summary_attn_bias=summary_attn_bias,
        summary_conditioner=summary_conditioner,
        summary_adapter=summary_adapter,
        summary_logit_bias=summary_logit_bias,
        prefix_generator=prefix_generator,
    )
    print(f"\n  CCT (sequential): loss={seq_loss:.4f}  ppl={seq_ppl:.2f}  ({seq_tokens} tokens, {seq_batches} batches)")

    if summary_conditioner is not None:
        summary_conditioner.remove_hooks()
    if summary_adapter is not None:
        summary_adapter.remove_hooks()
    del cct_model, commitment_head, up_project, summary_buffer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════
    delta_std_pct = (cct_std_ppl - bl_ppl) / bl_ppl * 100
    delta_mask_pct = (cct_mask_ppl - bl_ppl) / bl_ppl * 100
    delta_seq_pct = (seq_ppl - bl_ppl) / bl_ppl * 100

    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON")
    print("=" * 60)
    print(f"  Baseline PPL:             {bl_ppl:.2f}")
    print(f"  CCT (std attn) PPL:       {cct_std_ppl:.2f}  (delta: {delta_std_pct:+.2f}%)")
    print(f"  CCT (masked+summary) PPL: {cct_mask_ppl:.2f}  (delta: {delta_mask_pct:+.2f}%)")
    print(f"  CCT (sequential) PPL:     {seq_ppl:.2f}  (delta: {delta_seq_pct:+.2f}%)")
    print(f"  Target:                   < +10%")
    print(f"  Weights-only result:      {'PASS' if delta_std_pct < 10 else 'FAIL'}")
    print(f"  2-pass deploy result:     {'PASS' if delta_mask_pct < 10 else 'FAIL'}")
    print(f"  Sequential deploy result: {'PASS' if delta_seq_pct < 10 else 'FAIL'}")

    # Save results
    results = {
        "baseline": {
            "loss": bl_loss, "perplexity": bl_ppl,
            "tokens": bl_tokens, "batches": bl_batches,
        },
        "cct_standard_attention": {
            "loss": cct_std_loss, "perplexity": cct_std_ppl,
            "tokens": cct_std_tokens, "batches": cct_std_batches,
            "delta_pct": delta_std_pct,
            "pass": delta_std_pct < 10,
        },
        "cct_masked_deployment": {
            "loss": cct_mask_loss, "perplexity": cct_mask_ppl,
            "tokens": cct_mask_tokens, "batches": cct_mask_batches,
            "delta_pct": delta_mask_pct,
            "pass": delta_mask_pct < 10,
        },
        "cct_sequential_deployment": {
            "loss": seq_loss, "perplexity": seq_ppl,
            "tokens": seq_tokens, "batches": seq_batches,
            "delta_pct": delta_seq_pct,
            "pass": delta_seq_pct < 10,
        },
    }
    results_dir = Path(config.get("results_dir", "./results-tier2"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "perplexity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
