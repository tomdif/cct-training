"""
Training entry point.

Usage:
  python scripts/train.py --config configs/tier1_160m.yaml --mode cct
  python scripts/train.py --config configs/tier1_160m.yaml --mode baseline
"""

import argparse
import json
import random
import sys
import os
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import CCTTrainer, TrainConfig
from src.training.data_pipeline import extend_tokenizer, annotate_step_boundaries


class StreamingTextDataset(IterableDataset):
    """Streaming dataset that tokenizes and annotates step boundaries."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        seq_len: int,
        step_token_id: int,
        step_length: int,
        step_mode: str = "fixed",
        text_field: str = "text",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.step_token_id = step_token_id
        self.step_length = step_length
        self.step_mode = step_mode
        self.text_field = text_field

        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get(self.text_field, "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]

                # Annotate with STEP tokens
                annotated = annotate_step_boundaries(
                    chunk, self.step_token_id, self.step_mode, self.step_length
                )
                # Truncate to seq_len (STEP tokens add length)
                annotated = annotated[: self.seq_len]

                yield torch.tensor(annotated, dtype=torch.long)


class MultiplexingDataset(IterableDataset):
    """Interleaves multiple streaming datasets by ratio using deficit-based scheduling."""

    def __init__(self, datasets_with_ratios):
        """Args: datasets_with_ratios: list of (IterableDataset, float) tuples."""
        self.sources = datasets_with_ratios

    def __iter__(self):
        iters = [(iter(ds), ratio) for ds, ratio in self.sources]
        counters = [0.0] * len(iters)
        while True:
            for i in range(len(iters)):
                counters[i] += iters[i][1]
            idx = max(range(len(iters)), key=lambda i: counters[i])
            counters[idx] -= 1.0
            try:
                yield next(iters[idx][0])
            except StopIteration:
                iters[idx] = (iter(self.sources[idx][0]), iters[idx][1])
                yield next(iters[idx][0])


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(args):
    config_dict = load_config(args.config)

    # Set seed
    seed = config_dict.get("seed", 42)
    torch.manual_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Metal (MPS) backend")
    else:
        device = "cpu"
        print("WARNING: No GPU found, training on CPU (very slow)")

    # Load tokenizer and model
    model_name = config_dict["base_model"]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 on CUDA, float32 on MPS/CPU (MPS has limited bf16 support)
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn_impl = config_dict.get("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=model_dtype,
    ).to(device)

    # Extend tokenizer with STEP token
    step_token_id = extend_tokenizer(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print(f"STEP token ID: {step_token_id}")

    if args.mode == "baseline":
        print("\n=== BASELINE TRAINING (no CCT) ===")
        _train_baseline(model, tokenizer, config_dict, device)
    else:
        print("\n=== CCT TRAINING ===")
        _train_cct(model, tokenizer, step_token_id, config_dict, device, args)


def _train_cct(model, tokenizer, step_token_id, config_dict, device, args):
    """Run CCT training."""
    train_config = TrainConfig(
        d_model=config_dict["d_model"],
        d_summary=config_dict["d_summary"],
        d_bottleneck=config_dict["d_bottleneck"],
        step_length=config_dict["step_length"],
        total_steps=config_dict["total_steps"],
        batch_size=config_dict["batch_size"],
        seq_len=config_dict["seq_len"],
        learning_rate=config_dict["learning_rate"],
        weight_decay=config_dict.get("weight_decay", 0.01),
        warmup_steps=config_dict.get("warmup_steps", 1000),
        max_grad_norm=config_dict.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=config_dict.get("gradient_accumulation_steps", 1),
        phase1_end=config_dict["curriculum"]["phase1_end"],
        phase2_end=config_dict["curriculum"]["phase2_end"],
        phase3_end=config_dict["curriculum"]["phase3_end"],
        delta_start=config_dict.get("loss_weights", {}).get("delta_start", 0.0),
        delta_max=config_dict.get("loss_weights", {}).get("delta_max", 0.2),
        delta_taper=config_dict.get("loss_weights", {}).get("delta_taper", False),
        gamma_max=config_dict.get("loss_weights", {}).get("gamma_max", 0.3),
        alpha_end=config_dict.get("loss_weights", {}).get("alpha_end", 0.5),
        beta_max=config_dict.get("loss_weights", {}).get("beta_max", 0.5),
        aux_std_weight=config_dict.get("loss_weights", {}).get("aux_std_weight", 0.0),
        # Commitment head options
        use_tanh=config_dict.get("use_tanh", True),
        use_l2_norm=config_dict.get("use_l2_norm", True),
        noise_injection=config_dict.get("noise_injection", False),
        noise_sigma_start=config_dict.get("noise_sigma_start", 0.0),
        noise_sigma_end=config_dict.get("noise_sigma_end", 0.0),
        # Decoder options
        decoder_type=config_dict.get("decoder_type", "linear"),
        decoder_bottleneck=config_dict.get("decoder_bottleneck", None),
        # Multi-token summaries
        n_summary_tokens=config_dict.get("n_summary_tokens", 1),
        # L_conclusion
        conclusion_n_tokens=config_dict.get("conclusion_n_tokens", 10),
        # Summary attention bias
        summary_attn_bias=config_dict.get("summary_attn_bias", False),
        summary_attn_bias_init=config_dict.get("summary_attn_bias_init", 5.0),
        # Summary conditioning (FiLM + residual injection)
        summary_conditioning=config_dict.get("summary_conditioning", False),
        # Summary adapter (gated cross-attention, for frozen-model training)
        summary_adapter=config_dict.get("summary_adapter", False),
        adapter_d=config_dict.get("adapter_d", 64),
        freeze_base_model=config_dict.get("freeze_base_model", False),
        summary_logit_bias=config_dict.get("summary_logit_bias", False),
        logit_bias_hidden_dim=config_dict.get("logit_bias_hidden_dim", 256),
        # KV-cache prefix delivery
        use_kv_prefix=config_dict.get("use_kv_prefix", False),
        kv_prefix_hidden=config_dict.get("kv_prefix_hidden", 512),
        # Pseudo-token decoder
        use_pseudo_tokens=config_dict.get("use_pseudo_tokens", False),
        n_pseudo_tokens=config_dict.get("n_pseudo_tokens", 8),
        pseudo_decoder_hidden=config_dict.get("pseudo_decoder_hidden", 512),
        # LoRA
        use_lora=config_dict.get("use_lora", False),
        lora_rank=config_dict.get("lora_rank", 16),
        lora_alpha=config_dict.get("lora_alpha", 32),
        lora_dropout=config_dict.get("lora_dropout", 0.0),
        lora_target_modules=config_dict.get("lora_target_modules", ["query_key_value"]),
        lora_layers_min=config_dict.get("lora_layers_min", 12),
        lora_layers_max=config_dict.get("lora_layers_max", 23),
        lora_lr=config_dict.get("lora_lr", 2e-5),
        # Gradient isolation
        gradient_isolation=config_dict.get("gradient_isolation", "full_detach"),
        gradient_scale=config_dict.get("gradient_scale", 0.1),
        # Summary injection
        summary_injection_mode=config_dict.get("summary_injection", "replace"),
        # Training mode
        training_mode=config_dict.get("training_mode", "two_pass"),
        # Recurrent commitment head
        recurrent_commitment=config_dict.get("recurrent_commitment", False),
        last_summary_only=config_dict.get("last_summary_only", False),
        multi_hop_loss=config_dict.get("multi_hop_loss", False),
        multi_hop_weight=config_dict.get("multi_hop_weight", 0.25),
        synthetic_data_ratio=config_dict.get("synthetic_data_ratio", 0.0),
        eval_interval=config_dict.get("eval_interval", 5000),
        log_interval=config_dict.get("log_interval", 100),
        device=device,
        mixed_precision=config_dict.get("mixed_precision", "bf16"),
    )

    trainer = CCTTrainer(model, tokenizer, train_config, step_token_id)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"  Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  Resumed at step {trainer.global_step}")

    # Create dataset — multi-source mixing if code_dataset is specified
    ds_kwargs = dict(
        split=config_dict.get("dataset_split", "train"),
        tokenizer=tokenizer,
        seq_len=config_dict["seq_len"],
        step_token_id=step_token_id,
        step_length=config_dict["step_length"],
        step_mode=config_dict.get("step_boundary_mode", "fixed"),
    )
    code_ds_name = config_dict.get("code_dataset")
    if code_ds_name:
        code_ratio = config_dict.get("code_data_ratio", 0.7)
        code_field = config_dict.get("code_text_field", "content")
        web_field = config_dict.get("web_text_field", "text")
        web_dataset = StreamingTextDataset(
            dataset_name=config_dict["dataset"], text_field=web_field, **ds_kwargs)
        code_dataset = StreamingTextDataset(
            dataset_name=code_ds_name, text_field=code_field, **ds_kwargs)
        dataset = MultiplexingDataset([
            (code_dataset, code_ratio),
            (web_dataset, 1.0 - code_ratio),
        ])
        print(f"  Multi-source data: {code_ratio:.0%} code ({code_ds_name}) + {1-code_ratio:.0%} web ({config_dict['dataset']})")
    else:
        dataset = StreamingTextDataset(
            dataset_name=config_dict["dataset"],
            text_field=config_dict.get("web_text_field", "text"),
            **ds_kwargs,
        )
    dataloader = DataLoader(dataset, batch_size=config_dict["batch_size"])

    # Synthetic long-dependency data mixing
    synth_ratio = config_dict.get("synthetic_data_ratio", 0.0)
    synth_iter = None
    if synth_ratio > 0:
        from src.training.synthetic_data import SyntheticLongDepDataset
        synth_dataset = SyntheticLongDepDataset(
            dataset_name=config_dict["dataset"],
            split=config_dict.get("dataset_split", "train"),
            tokenizer=tokenizer,
            seq_len=config_dict["seq_len"],
            step_token_id=step_token_id,
            step_length=config_dict["step_length"],
            multi_marker=config_dict.get("multi_marker", False),
        )
        synth_loader = DataLoader(synth_dataset, batch_size=config_dict["batch_size"])
        synth_iter = iter(synth_loader)
        print(f"  Synthetic data mixing enabled: {synth_ratio:.0%} ratio")

    # Training loop
    save_dir = Path(config_dict.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = False
    try:
        import wandb
        wandb.init(
            project=config_dict.get("wandb_project", "cct-training"),
            name=config_dict.get("wandb_name", "cct-run"),
            config=config_dict,
        )
        use_wandb = True
    except Exception:
        print("wandb not available, logging to stdout only")

    resume_step = trainer.global_step
    remaining = train_config.total_steps - resume_step
    if resume_step > 0:
        print(f"\nResuming training from step {resume_step}, {remaining} steps remaining...")
    else:
        print(f"\nTraining for {train_config.total_steps} steps...")
    t0 = time.time()
    batch_count = 0
    print("  Waiting for first batch from dataloader...", flush=True)

    for batch in dataloader:
        if trainer.global_step >= train_config.total_steps:
            break

        batch_count += 1

        # Mix synthetic data with configurable probability
        if synth_iter is not None and random.random() < synth_ratio:
            try:
                batch = next(synth_iter)
            except StopIteration:
                synth_iter = iter(synth_loader)
                batch = next(synth_iter)

        if batch_count <= 3 or (batch_count <= 100 and batch_count % 10 == 0):
            print(f"  [batch {batch_count}] step={trainer.global_step} shape={batch.shape} t={time.time()-t0:.1f}s", flush=True)

        loss_dict = trainer.train_step(batch)

        # Log
        if trainer.global_step % train_config.log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = (trainer.global_step - resume_step) / elapsed if elapsed > 0 else 0
            total_val = loss_dict['total'].item() if hasattr(loss_dict['total'], 'item') else loss_dict['total']
            log_line = (
                f"  step {trainer.global_step:>6} | "
                f"loss {total_val:.4f} | "
                f"phase {loss_dict.get('phase', '?')} | "
                f"p_commit {loss_dict.get('p_commit', 0):.2f} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            # Log individual loss components for CCT steps
            if loss_dict.get('n_summaries', 0) > 0:
                log_line += (
                    f" | L_std {loss_dict.get('standard', 0):.4f}"
                    f" L_val {loss_dict.get('validity', 0):.4f}"
                    f" L_con {loss_dict.get('conclusion', 0):.4f}"
                    f" L_suf {loss_dict.get('sufficiency', 0):.4f}"
                    f" n_sum {loss_dict.get('n_summaries', 0)}"
                )
                if 'gate_mean' in loss_dict:
                    log_line += f" gate {loss_dict['gate_mean']:.4f}"
                if 'hop' in loss_dict:
                    log_line += f" L_hop {loss_dict['hop']:.4f}"
                if 'adapter_gate_mean' in loss_dict:
                    log_line += f" adpt {loss_dict['adapter_gate_mean']:.3f}/{loss_dict['adapter_gate_max']:.3f}"
            print(log_line)
            if use_wandb:
                wandb.log(loss_dict, step=trainer.global_step)

        # Save checkpoint
        save_interval = config_dict.get("save_interval", 10000)
        if trainer.global_step % save_interval == 0 and trainer.global_step > 0:
            ckpt_path = save_dir / f"cct-{trainer.global_step}.pt"
            trainer.save_checkpoint(str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = save_dir / "cct-final.pt"
    trainer.save_checkpoint(str(final_path))
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Total time: {time.time() - t0:.0f}s")

    if use_wandb:
        wandb.finish()


def _train_baseline(model, tokenizer, config_dict, device):
    """Standard fine-tuning without CCT (baseline comparison)."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    total_steps = config_dict.get("baseline_steps", config_dict["total_steps"])
    lr = config_dict["learning_rate"]
    batch_size = config_dict["batch_size"]
    seq_len = config_dict["seq_len"]

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Use the same STEP token ID for consistent data
    step_token_id = tokenizer.convert_tokens_to_ids("<STEP>")

    dataset = StreamingTextDataset(
        dataset_name=config_dict["dataset"],
        split=config_dict.get("dataset_split", "train"),
        tokenizer=tokenizer,
        seq_len=seq_len,
        step_token_id=step_token_id,
        step_length=config_dict["step_length"],
        step_mode=config_dict.get("step_boundary_mode", "fixed"),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    save_dir = Path(config_dict.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaseline training for {total_steps} steps...")
    t0 = time.time()
    global_step = 0

    model.train()
    for batch in dataloader:
        if global_step >= total_steps:
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
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        if global_step % config_dict.get("log_interval", 100) == 0:
            print(f"  step {global_step:>6} | loss {loss.item():.4f}")

    # Save
    model.save_pretrained(save_dir / "baseline-final")
    tokenizer.save_pretrained(save_dir / "baseline-final")
    print(f"\nBaseline training complete. Saved to {save_dir / 'baseline-final'}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCT Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["cct", "baseline"], default="cct")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)
