"""
Evaluation entry point.

Usage:
  python scripts/evaluate.py --checkpoint ./checkpoints/cct-final.pt --config configs/tier1_160m.yaml
  python scripts/evaluate.py --baseline ./checkpoints/baseline-final --config configs/tier1_160m.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.commitment_head import CommitmentHead
from src.training.data_pipeline import extend_tokenizer
from src.evaluation.sufficiency import evaluate_sufficiency
from src.evaluation.compression import measure_compression, print_compression_table
from src.evaluation.passkey_retrieval import evaluate_passkey_retrieval


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(args):
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_token_id = extend_tokenizer(tokenizer)

    results = {"config": config}

    # === Compression (always, no model needed) ===
    print("\n=== COMPRESSION MEASUREMENT ===")
    compression = measure_compression(
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        d_summary=config["d_summary"],
        step_length=config["step_length"],
        context_lengths=config.get(
            "passkey_context_lengths", [512, 1024, 2048, 4096]
        ),
    )
    print_compression_table(compression, model_name)
    results["compression"] = {
        str(k): v for k, v in compression.items()
    }

    # === Load CCT model and components ===
    if args.checkpoint:
        print(f"\nLoading CCT checkpoint: {args.checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).to(device)
        model.resize_token_embeddings(len(tokenizer))

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        commitment_head = CommitmentHead(
            d_model=config["d_model"],
            d_summary=config["d_summary"],
            d_bottleneck=config["d_bottleneck"],
        ).to(device)
        commitment_head.load_state_dict(ckpt["commitment_head_state_dict"])

        # === Sufficiency Probe ===
        print("\n=== SUFFICIENCY PROBE ===")
        # Build a small eval dataloader
        from src.evaluation.sufficiency import (
            collect_summary_hidden_pairs,
            train_linear_probe,
        )
        from src.training.data_pipeline import annotate_step_boundaries
        from torch.utils.data import DataLoader, IterableDataset
        from scripts.train import StreamingTextDataset

        eval_dataset = StreamingTextDataset(
            dataset_name=config.get("validation_dataset", config["dataset"]),
            split=config.get("validation_split", "validation"),
            tokenizer=tokenizer,
            seq_len=config["seq_len"],
            step_token_id=step_token_id,
            step_length=config["step_length"],
        )
        eval_loader = DataLoader(eval_dataset, batch_size=4)

        suff_results = evaluate_sufficiency(
            model, commitment_head, eval_loader, step_token_id, device
        )
        print(f"  Sufficiency R²: {suff_results['sufficiency_r2']:.4f}")
        print(f"  Pairs collected: {suff_results['n_pairs']}")
        print(f"  Compression: {suff_results['d_model']}d -> {suff_results['d_summary']}d")
        results["sufficiency"] = suff_results

        # === Passkey Retrieval ===
        print("\n=== PASSKEY RETRIEVAL ===")
        passkey_lengths = config.get("passkey_context_lengths", [512, 1024, 2048])
        passkey_results = evaluate_passkey_retrieval(
            model, tokenizer, passkey_lengths,
            step_token_id=step_token_id,
            step_length=config["step_length"],
            device=device,
        )
        for ctx_len, acc in sorted(passkey_results.items()):
            print(f"  Context {ctx_len:>6}: {acc:.1%}")
        results["passkey"] = {str(k): v for k, v in passkey_results.items()}

    # === Check success criteria ===
    criteria = config.get("success_criteria", {})
    if criteria and "sufficiency" in results:
        print("\n=== SUCCESS CRITERIA ===")
        r2 = results["sufficiency"]["sufficiency_r2"]
        r2_target = criteria.get("sufficiency_probe_r2", 0.85)
        r2_pass = r2 >= r2_target
        print(f"  Sufficiency R² {r2:.4f} >= {r2_target} : {'PASS' if r2_pass else 'FAIL'}")

        for ctx_str, ratio_info in results.get("compression", {}).items():
            ctx = int(ctx_str)
            key = f"compression_ratio_{ctx // 1000}k"
            if key in criteria:
                actual = ratio_info["compression_ratio"]
                target = criteria[key]
                passed = actual >= target
                print(f"  Compression @{ctx}: {actual:.0f}x >= {target}x : {'PASS' if passed else 'FAIL'}")

    # Save results
    results_dir = Path(config.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCT Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", help="Path to CCT checkpoint (.pt)")
    parser.add_argument("--baseline", help="Path to baseline model dir")
    args = parser.parse_args()
    evaluate(args)
