"""
Summary quantization sweep: test INT8/INT4/INT2 compression of summary vectors.

Takes an existing CCT checkpoint and evaluates hybrid PPL with quantized
summaries. Quantization happens after the commitment head, before pseudo-token
decoding. This tests whether the 128-dim summary vectors can be aggressively
compressed without losing information.

Patent value: "summary bank of 12KB for 100K tokens" (INT4) or 6KB (INT2).

Usage:
  python scripts/eval_quantization.py \
    --config configs/tier3_410m_retrieval_P.yaml \
    --cct-checkpoint checkpoints-run-P-retrieval/cct-final.pt \
    --eval-batches 100 --seq-len 2048
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
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
from src.model.pseudo_token_decoder import PseudoTokenDecoder
from src.model.kv_cache_utils import (
    _cache_to_tuples, _tuples_to_cache, extract_step_kv, merge_kv_caches,
)


# ══════════════════════════════════════════════════════════
# Quantization functions
# ══════════════════════════════════════════════════════════

def quantize_summary(summary, bits):
    """Symmetric uniform quantization of summary vectors.

    Args:
        summary: (batch, d_summary) or (batch, K, d_summary) float tensor
        bits: number of bits (8, 4, 2, 1)

    Returns:
        Dequantized float tensor (simulates quantize → store → dequantize)
    """
    if bits >= 32:
        return summary  # No quantization

    n_levels = 2 ** bits
    qmin = -(n_levels // 2)
    qmax = n_levels // 2 - 1

    # Per-vector symmetric quantization (scale per summary vector)
    abs_max = summary.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / qmax

    # Quantize
    quantized = (summary / scale).round().clamp(qmin, qmax)

    # Dequantize
    dequantized = quantized * scale

    return dequantized


def summary_size_bytes(d_summary, bits, n_summaries):
    """Calculate storage for n_summaries at given bit width."""
    return n_summaries * d_summary * bits / 8


# ══════════════════════════════════════════════════════════
# Streaming eval dataset
# ══════════════════════════════════════════════════════════

class StreamingEvalDataset(torch.utils.data.IterableDataset):
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


# ══════════════════════════════════════════════════════════
# Eval with quantized summaries
# ══════════════════════════════════════════════════════════

def eval_quantized(
    model, commitment_head, pseudo_decoder, embed_layer,
    dataloader, step_token_id, device, max_batches,
    n_summary_tokens_K=1,
    retrieval_k=2,
    max_bank_size=8,
    quant_bits=32,
):
    """Hybrid eval with summary quantization at specified bit width."""
    model.eval()
    commitment_head.eval()
    if pseudo_decoder is not None:
        pseudo_decoder.eval()

    step_loss = defaultdict(float)
    step_tokens = defaultdict(int)
    n_batches = 0
    n_layers = model.config.num_hidden_layers
    K = n_summary_tokens_K

    # Track quantization error
    quant_errors = []

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
            kv_bank = []

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
                n_prepend = 0
                retrieved_kv = None

                # ---- Pseudo-token prefix from quantized summaries ----
                if (pseudo_decoder is not None and n_prior_steps > 0):
                    P = pseudo_decoder.n_pseudo_tokens
                    n_prepend = n_prior_steps * P
                    summary_embeds_list = []
                    summary_pos_list = []

                    for s_idx in range(n_prior_steps):
                        s = committed_summaries[s_idx]
                        if K > 1:
                            s = s[:, 0, :]
                        decoded = pseudo_decoder(s).to(step_embeds.dtype)
                        summary_embeds_list.append(decoded)

                    for i in range(n_prepend):
                        summary_pos_list.append(max(0, start - n_prepend + i))

                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)
                    summary_positions = torch.tensor(
                        summary_pos_list, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([summary_positions, step_positions], dim=1)
                else:
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                # ---- Retrieved KV cache ----
                if len(kv_bank) > 0:
                    k_ret = min(retrieval_k, len(kv_bank))
                    start_idx = max(0, len(kv_bank) - k_ret)
                    indices = list(range(start_idx, len(kv_bank)))
                    selected_caches = [kv_bank[i] for i in indices]
                    retrieved_kv = merge_kv_caches(selected_caches, n_layers)

                # ---- Forward pass ----
                past_kv_arg = (
                    _tuples_to_cache(retrieved_kv)
                    if retrieved_kv is not None else None
                )
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    past_key_values=past_kv_arg,
                    use_cache=True,
                )

                # ---- Store KV ----
                is_last_step = (step_idx == len(step_ranges) - 1)
                if not is_last_step and outputs.past_key_values is not None:
                    step_kv = extract_step_kv(
                        outputs.past_key_values, step_len, n_layers
                    )
                    kv_bank.append(step_kv)
                    while len(kv_bank) > max_bank_size:
                        evicted = kv_bank.pop(0)
                        del evicted

                # ---- Commitment summary with quantization ----
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][:, n_prepend:, :]
                    prev_sum = (
                        committed_summaries[-1] if committed_summaries else None
                    )
                    summary = commitment_head(
                        step_hidden.float(), prev_summary=prev_sum
                    )

                    # QUANTIZE HERE
                    if quant_bits < 32:
                        original = summary.clone()
                        summary = quantize_summary(summary, quant_bits)
                        # Track per-vector L2 error
                        err = (summary - original).pow(2).sum(-1).sqrt().mean().item()
                        quant_errors.append(err)

                    committed_summaries.append(summary)

                # ---- Loss computation ----
                step_logits = outputs.logits[
                    :, n_prepend:n_prepend + step_len - 1, :
                ]
                step_labels = step_token_ids[:, 1:].clone()
                step_labels[step_labels == step_token_id] = -100

                if n_prepend > 0:
                    cross_logit = outputs.logits[
                        :, n_prepend - 1:n_prepend, :
                    ]
                    first_label = step_token_ids[:, 0:1].clone()
                    first_label[first_label == step_token_id] = -100
                    step_logits = torch.cat([cross_logit, step_logits], dim=1)
                    step_labels = torch.cat([first_label, step_labels], dim=1)

                loss = F.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    step_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (step_labels.reshape(-1) != -100).sum().item()
                step_loss[step_idx] += loss.item()
                step_tokens[step_idx] += n_valid

            kv_bank.clear()

            n_batches += 1
            if n_batches % 50 == 0:
                total_l = sum(step_loss.values())
                total_t = sum(step_tokens.values())
                avg = total_l / total_t if total_t > 0 else 0
                print(f"  batch {n_batches}/{max_batches} | "
                      f"avg loss {avg:.4f} | ppl {math.exp(avg):.2f}")

    avg_quant_err = sum(quant_errors) / len(quant_errors) if quant_errors else 0.0
    return step_loss, step_tokens, n_batches, avg_quant_err


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Summary quantization sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--bits", type=str, default="32,8,4,2",
                        help="Comma-separated bit widths to test")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seq_len = args.seq_len or config["seq_len"]
    step_length = config["step_length"]
    d_summary = config["d_summary"]
    n_summary_tokens = config.get("n_summary_tokens", 1)
    bit_widths = [int(b) for b in args.bits.split(",")]

    print(f"Config: seq_len={seq_len}, step_length={step_length}, d_summary={d_summary}")
    print(f"Bit widths to test: {bit_widths}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found, running on CPU")

    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_token_id = extend_tokenizer(tokenizer)

    batch_size = config.get("batch_size", 2)

    print("\nLoading model...")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn_impl = config.get("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=model_dtype,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)

    cct_state = {
        k: v.to(model_dtype) if v.is_floating_point() else v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(cct_state)

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

    pseudo_decoder = None
    if ckpt.get("pseudo_decoder_state_dict") is not None:
        pseudo_decoder = PseudoTokenDecoder(
            d_summary=config["d_summary"],
            d_model=config["d_model"],
            n_pseudo_tokens=config.get("n_pseudo_tokens", 8),
            hidden_dim=config.get("pseudo_decoder_hidden", 512),
            device=device,
        )
        pseudo_decoder.load_state_dict(ckpt["pseudo_decoder_state_dict"])
        pseudo_decoder.eval()

    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.gpt_neox.embed_in

    retrieval_k = config.get("retrieval_k", 2)
    max_bank_size = config.get("max_kv_bank_size", 8)

    def make_loader():
        ds = StreamingEvalDataset(
            dataset_name=config.get("validation_dataset", config["dataset"]),
            split=config.get("validation_split", "train"),
            tokenizer=tokenizer, seq_len=seq_len,
            step_token_id=step_token_id, step_length=step_length,
            skip_examples=50000,
        )
        return DataLoader(ds, batch_size=batch_size)

    # ── Run sweep ──
    results = {}
    steps_per_seq = seq_len // (step_length + 1)

    print("\n" + "=" * 80)
    print("SUMMARY QUANTIZATION SWEEP")
    print("=" * 80)

    for bits in bit_widths:
        label = f"FP32" if bits >= 32 else f"INT{bits}"
        print(f"\n{'─' * 60}")
        print(f"  {label} ({bits}-bit summaries)")
        print(f"{'─' * 60}")

        # Storage calculation
        # At 100K tokens: ~250 steps of 400 tokens, each summary is d_summary dims
        n_summaries_100k = 100000 // step_length
        storage = summary_size_bytes(d_summary, bits, n_summaries_100k)
        print(f"  Storage for 100K tokens: {storage / 1024:.1f} KB "
              f"({n_summaries_100k} summaries x {d_summary} dims x {bits} bits)")

        s_loss, s_tok, n_batch, avg_err = eval_quantized(
            model, commitment_head, pseudo_decoder, embed_layer,
            make_loader(), step_token_id, device, args.eval_batches,
            n_summary_tokens_K=n_summary_tokens,
            retrieval_k=retrieval_k,
            max_bank_size=max_bank_size,
            quant_bits=bits,
        )

        total_loss = sum(s_loss.values())
        total_tok = sum(s_tok.values())
        avg_loss = total_loss / total_tok if total_tok > 0 else 0
        ppl = math.exp(avg_loss)

        results[bits] = {
            "ppl": ppl,
            "avg_loss": avg_loss,
            "quant_error_l2": avg_err,
            "storage_100k_kb": storage / 1024,
            "per_step": {
                str(si): math.exp(s_loss[si] / s_tok[si]) if s_tok.get(si, 0) > 0 else None
                for si in sorted(s_loss.keys())
            },
        }

        print(f"  PPL: {ppl:.2f} | avg quant error: {avg_err:.6f}")

    # ── Summary table ──
    print("\n\n" + "=" * 80)
    print("QUANTIZATION SWEEP RESULTS")
    print("=" * 80)
    print(f"  {'Bits':>6}  {'Label':>6}  {'PPL':>8}  {'Delta':>8}  "
          f"{'Quant Err':>10}  {'100K Store':>10}  {'Compression':>12}")
    print("-" * 80)

    baseline_ppl = results.get(32, {}).get("ppl", None)
    for bits in bit_widths:
        r = results[bits]
        label = "FP32" if bits >= 32 else f"INT{bits}"
        delta = r["ppl"] - baseline_ppl if baseline_ppl else 0
        storage_kb = r["storage_100k_kb"]
        # Compression vs FP32 storage
        fp32_kb = summary_size_bytes(d_summary, 32, 100000 // step_length) / 1024
        compression = fp32_kb / storage_kb if storage_kb > 0 else 1.0

        print(f"  {bits:>6}  {label:>6}  {r['ppl']:>8.2f}  {delta:>+8.2f}  "
              f"{r['quant_error_l2']:>10.6f}  {storage_kb:>8.1f} KB  {compression:>10.1f}x")

    # ── Per-step breakdown ──
    print(f"\n  {'Step':>4}", end="")
    for bits in bit_widths:
        label = "FP32" if bits >= 32 else f"INT{bits}"
        print(f"  {label:>8}", end="")
    print()
    print("-" * (8 + 10 * len(bit_widths)))

    all_steps = sorted(set().union(*(results[b]["per_step"].keys() for b in bit_widths)))
    for si in all_steps:
        print(f"  {si:>4}", end="")
        for bits in bit_widths:
            v = results[bits]["per_step"].get(si, None)
            if v is not None:
                print(f"  {v:>8.2f}", end="")
            else:
                print(f"  {'n/a':>8}", end="")
        print()

    # ── Patent-relevant summary ──
    print("\n" + "=" * 80)
    print("PATENT-RELEVANT FINDINGS")
    print("=" * 80)
    for bits in [8, 4, 2]:
        if bits in results and baseline_ppl:
            r = results[bits]
            delta = r["ppl"] - baseline_ppl
            storage_kb = r["storage_100k_kb"]
            n_summ = 100000 // step_length
            label = f"INT{bits}"
            status = "PASS" if abs(delta) < 0.5 else ("MARGINAL" if abs(delta) < 1.0 else "FAIL")
            print(f"  {label}: {r['ppl']:.2f} PPL ({delta:+.2f} vs FP32) | "
                  f"{storage_kb:.1f} KB for 100K tokens ({n_summ} summaries) | {status}")

    # ── Save results ──
    results_dir = Path(config.get("results_dir", "./results-quantization"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"quantization_sweep_seqlen{seq_len}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
