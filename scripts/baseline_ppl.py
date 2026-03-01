"""Compute full-attention baseline PPL for the base model (no CCT)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--batches", type=int, default=50)
args = parser.parse_args()

print(f"Loading {args.model}...")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model.eval()

print(f"Evaluating {args.batches} batches at seq_len={args.seq_len}...")
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
total_loss, total_tokens, count = 0.0, 0, 0

with torch.no_grad():
    for i, example in enumerate(ds):
        if count >= args.batches:
            break
        ids = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids.cuda()
        if ids.shape[1] < 512:
            continue
        out = model(ids, labels=ids)
        total_loss += out.loss.item() * (ids.shape[1] - 1)
        total_tokens += ids.shape[1] - 1
        count += 1
        if count % 10 == 0:
            print(f"  batch {count}/{args.batches}: running PPL = {math.exp(total_loss/total_tokens):.2f}")

ppl = math.exp(total_loss / total_tokens)
print(f"\n{args.model} full-attention baseline PPL ({args.seq_len}): {ppl:.2f}")
