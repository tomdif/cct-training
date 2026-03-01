"""
Hierarchical Summary Tree Eval

Tests whether organizing summaries into a tree structure (instead of flat
GRU chain) fixes the degradation at 32K+ tokens.

Architecture:
  Level 0: Per-step summaries from commitment head (existing, no GRU recurrence)
  Level 1: Mean-pool groups of B level-0 summaries
  Level 2: Mean-pool groups of B level-1 summaries
  Level 3: Mean-pool groups of B level-2 summaries (if needed)

At each step, pseudo-tokens are decoded from a multi-level selection:
  - Recent level-0 summaries (local detail)
  - Covering level-1 summary(ies) (section context)
  - Level-2+ summary (document context)

This gives O(log N) depth instead of O(N) sequential GRU updates.

Comparison conditions:
  1. Flat GRU (current):   decode all GRU-accumulated summaries -> pseudo-tokens
  2. Hierarchical tree:    decode multi-level selection -> pseudo-tokens
  3. Sliding window:       800 tokens raw context (baseline)

Usage:
  python scripts/eval_hierarchical.py \
    --config configs/tier3_410m_retrieval_P.yaml \
    --cct-checkpoint checkpoints-run-P-retrieval/cct-final.pt \
    --seq-len 16384 --branching-factor 8
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_pipeline import (
    extend_tokenizer,
    annotate_step_boundaries,
    get_step_boundary_positions,
    get_step_ranges,
)
from src.model.commitment_head import CommitmentHead
from src.model.pseudo_token_decoder import PseudoTokenDecoder
from src.model.kv_cache_utils import (
    _cache_to_tuples, _tuples_to_cache, extract_step_kv, merge_kv_caches,
)


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


def build_summary_tree(summaries, branching_factor=8):
    """Build hierarchical tree from flat list of level-0 summaries.

    Args:
        summaries: list of tensors, each (batch, d_summary)
        branching_factor: how many children per parent node

    Returns:
        tree: dict mapping level -> list of summary tensors
              level 0 = raw summaries, level 1 = mean of groups, etc.
    """
    tree = {0: summaries}
    level = 0
    current = summaries

    while len(current) > 1:
        level += 1
        parents = []
        for i in range(0, len(current), branching_factor):
            group = current[i:i + branching_factor]
            if len(group) == 0:
                continue
            # Mean-pool the group
            stacked = torch.stack(group, dim=0)  # (group_size, batch, d_summary)
            parent = stacked.mean(dim=0)  # (batch, d_summary)
            # Re-normalize to unit sphere (summaries are L2-normalized)
            parent = F.normalize(parent, dim=-1)
            parents.append(parent)
        tree[level] = parents
        current = parents

    return tree


def select_from_tree(tree, current_step, n_recent=3, branching_factor=8):
    """Select summaries from the tree for pseudo-token decoding.

    Strategy:
      - n_recent most recent level-0 summaries (local detail)
      - The level-1 summary covering the current position
      - The level-2+ summaries (broader context)
      - Skip any summaries that are "current" (would be cheating)

    Args:
        tree: dict from build_summary_tree
        current_step: which step we're about to process (0-indexed)
        n_recent: how many recent level-0 summaries to include
        branching_factor: must match build_summary_tree

    Returns:
        selected: list of summary tensors to decode into pseudo-tokens
    """
    selected = []
    max_level = max(tree.keys())

    # Level 0: n_recent most recent summaries (before current_step)
    level0 = tree[0]
    available_l0 = level0[:current_step]  # only summaries before current step
    recent = available_l0[-n_recent:] if len(available_l0) >= n_recent else available_l0
    selected.extend(recent)
    recent_indices = set(range(max(0, current_step - n_recent), current_step))

    # Higher levels: include summaries that cover steps NOT already in recent
    for level in range(1, max_level + 1):
        level_summaries = tree[level]
        group_size = branching_factor ** level  # each level-L summary covers B^L steps

        for idx, summary in enumerate(level_summaries):
            # What step range does this summary cover?
            cover_start = idx * group_size
            cover_end = min((idx + 1) * group_size, current_step)

            if cover_end <= 0:
                continue  # nothing relevant
            if cover_start >= current_step:
                continue  # future summary, skip

            # Check if this summary's coverage is already fully in recent
            covered_steps = set(range(cover_start, cover_end))
            if covered_steps.issubset(recent_indices):
                continue  # already have detailed level-0 for all these steps

            selected.append(summary)

    return selected


def eval_condition(
    model, commitment_head, pseudo_decoder, embed_layer,
    dataloader, step_token_id, device, max_batches,
    mode="flat_gru",  # "flat_gru", "hierarchical", "isolated"
    retrieval_k=2, max_bank_size=8,
    branching_factor=8, n_recent_l0=3,
    config=None,
):
    """Run one eval condition with per-step PPL tracking.

    Modes:
      flat_gru:      Current approach — GRU recurrence, decode all summaries
      hierarchical:  Tree structure — no GRU, multi-level selection
      isolated:      No context at all (cold start baseline)
    """
    model.eval()
    commitment_head.eval()
    if pseudo_decoder is not None:
        pseudo_decoder.eval()

    step_loss = defaultdict(float)
    step_tokens = defaultdict(int)
    n_batches = 0
    n_layers = model.config.num_hidden_layers

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

            # Accumulate summaries for all steps
            gru_summaries = []       # GRU-accumulated (for flat_gru mode)
            raw_summaries = []       # Non-recurrent (for hierarchical mode)
            kv_bank = []
            summary_tree = None

            for step_idx, (start, end) in enumerate(step_ranges):
                step_len = end - start
                if step_len <= 0:
                    continue

                step_token_ids = input_ids[:, start:end]
                step_embeds = embed_layer(step_token_ids)
                step_positions = torch.arange(
                    start, end, device=device
                ).unsqueeze(0).expand(batch_size, -1)

                n_prepend = 0
                retrieved_kv = None
                is_last_step = (step_idx == len(step_ranges) - 1)

                # ---- Build pseudo-token prefix based on mode ----
                if mode == "flat_gru" and pseudo_decoder is not None and len(gru_summaries) > 0:
                    # Current approach: decode ALL GRU summaries
                    P = pseudo_decoder.n_pseudo_tokens
                    n_prepend = len(gru_summaries) * P
                    summary_embeds_list = []
                    for s in gru_summaries:
                        decoded = pseudo_decoder(s).to(step_embeds.dtype)
                        summary_embeds_list.append(decoded)
                    summary_embeds = torch.cat(summary_embeds_list, dim=1)
                    inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)

                    summary_pos = [max(0, start - n_prepend + i) for i in range(n_prepend)]
                    summary_positions = torch.tensor(
                        summary_pos, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([summary_positions, step_positions], dim=1)

                elif mode == "hierarchical" and pseudo_decoder is not None and len(raw_summaries) > 0:
                    # Hierarchical: build tree, select multi-level
                    summary_tree = build_summary_tree(raw_summaries, branching_factor)
                    selected = select_from_tree(
                        summary_tree, step_idx,
                        n_recent=n_recent_l0,
                        branching_factor=branching_factor,
                    )

                    if len(selected) > 0:
                        P = pseudo_decoder.n_pseudo_tokens
                        n_prepend = len(selected) * P
                        summary_embeds_list = []
                        for s in selected:
                            decoded = pseudo_decoder(s).to(step_embeds.dtype)
                            summary_embeds_list.append(decoded)
                        summary_embeds = torch.cat(summary_embeds_list, dim=1)
                        inputs_embeds = torch.cat([summary_embeds, step_embeds], dim=1)

                        summary_pos = [max(0, start - n_prepend + i) for i in range(n_prepend)]
                        summary_positions = torch.tensor(
                            summary_pos, dtype=torch.long, device=device
                        ).unsqueeze(0).expand(batch_size, -1)
                        position_ids = torch.cat([summary_positions, step_positions], dim=1)
                    else:
                        inputs_embeds = step_embeds
                        position_ids = step_positions

                else:
                    # Isolated or no summaries yet
                    inputs_embeds = step_embeds
                    position_ids = step_positions

                # ---- Retrieved KV (for flat_gru and hierarchical) ----
                if mode != "isolated" and len(kv_bank) > 0:
                    k = min(retrieval_k, len(kv_bank))
                    selected_caches = kv_bank[-k:]
                    retrieved_kv = merge_kv_caches(selected_caches, n_layers)

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
                if not is_last_step and outputs.past_key_values is not None:
                    step_kv = extract_step_kv(
                        outputs.past_key_values, step_len, n_layers
                    )
                    kv_bank.append(step_kv)
                    while len(kv_bank) > max_bank_size:
                        kv_bank.pop(0)

                # ---- Commitment summaries ----
                if not is_last_step:
                    step_hidden = outputs.hidden_states[-1][:, n_prepend:, :]

                    # GRU summary (for flat_gru mode)
                    prev_sum = gru_summaries[-1] if gru_summaries else None
                    gru_summary = commitment_head(
                        step_hidden.float(), prev_summary=prev_sum
                    )
                    gru_summaries.append(gru_summary)

                    # Raw summary WITHOUT GRU recurrence (for hierarchical mode)
                    raw_summary = commitment_head(
                        step_hidden.float(), prev_summary=None
                    )
                    raw_summaries.append(raw_summary)

                # ---- Loss ----
                step_logits = outputs.logits[:, n_prepend:n_prepend + step_len - 1, :]
                step_labels = step_token_ids[:, 1:].clone()
                step_labels[step_labels == step_token_id] = -100

                if n_prepend > 0:
                    cross_logit = outputs.logits[:, n_prepend - 1:n_prepend, :]
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

            # Clear between sequences
            kv_bank.clear()
            gru_summaries.clear()
            raw_summaries.clear()

            n_batches += 1
            if n_batches % 10 == 0:
                total_l = sum(step_loss.values())
                total_t = sum(step_tokens.values())
                avg = total_l / total_t if total_t > 0 else 0
                print(f"  [{mode}] batch {n_batches}/{max_batches} | "
                      f"ppl {math.exp(avg):.2f}")

    return step_loss, step_tokens, n_batches


def sliding_window_eval(model, embed_layer, tokenizer, dataloader,
                        step_token_id, device, max_batches, window_tokens=800,
                        step_length=400):
    """Sliding window baseline (prior art)."""
    model.eval()
    step_loss = defaultdict(float)
    step_tokens_dict = defaultdict(int)
    n_batches = 0

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

            for step_idx, (start, end) in enumerate(step_ranges):
                step_len = end - start
                if step_len <= 0:
                    continue

                window_start = max(0, start - window_tokens)
                context_ids = input_ids[:, window_start:end].clone()
                context_ids[context_ids == step_token_id] = 1

                ctx_len = context_ids.shape[1]
                position_ids = torch.arange(ctx_len, device=device).unsqueeze(0).expand(batch_size, -1)

                outputs = model(input_ids=context_ids, position_ids=position_ids)

                offset = start - window_start
                logits = outputs.logits[:, offset:offset + step_len - 1, :]
                labels = input_ids[:, start + 1:end].clone()
                labels[labels == step_token_id] = -100

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_valid = (labels.reshape(-1) != -100).sum().item()
                step_loss[step_idx] += loss.item()
                step_tokens_dict[step_idx] += n_valid

            n_batches += 1
            if n_batches % 10 == 0:
                total_l = sum(step_loss.values())
                total_t = sum(step_tokens_dict.values())
                avg = total_l / total_t if total_t > 0 else 0
                print(f"  [sliding_window] batch {n_batches}/{max_batches} | "
                      f"ppl {math.exp(avg):.2f}")

    return step_loss, step_tokens_dict, n_batches


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Summary Tree Eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cct-checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--branching-factor", type=int, default=8)
    parser.add_argument("--n-recent", type=int, default=3,
                        help="Number of recent level-0 summaries to include")
    parser.add_argument("--retrieval-k", type=int, default=2)
    parser.add_argument("--max-bank-size", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Branching factor: {args.branching_factor}")
    print(f"Recent level-0: {args.n_recent}")

    # Load model
    print(f"Loading model: {config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "eager"),
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    step_token_id = extend_tokenizer(tokenizer)

    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.gpt_neox.embed_in

    commitment_head = CommitmentHead(
        d_model=config["d_model"],
        d_summary=config.get("d_summary", 16),
        d_bottleneck=config.get("d_bottleneck", 64),
        n_summary_tokens=config.get("n_summary_tokens", 1),
        use_tanh=config.get("use_tanh", False),
        use_l2_norm=config.get("use_l2_norm", False),
        recurrent=config.get("recurrent_commitment", False),
    ).to(device)

    pseudo_decoder = None
    if config.get("use_pseudo_tokens", False):
        pseudo_decoder = PseudoTokenDecoder(
            d_summary=config.get("d_summary", 16),
            d_model=config["d_model"],
            n_pseudo_tokens=config.get("n_pseudo_tokens", 8),
            hidden_dim=config.get("pseudo_decoder_hidden", 512),
        ).to(device)

    ckpt = torch.load(args.cct_checkpoint, map_location=device, weights_only=False)
    commitment_head.load_state_dict(ckpt["commitment_head_state_dict"])
    if pseudo_decoder is not None and "pseudo_decoder_state_dict" in ckpt:
        pseudo_decoder.load_state_dict(ckpt["pseudo_decoder_state_dict"])
    print(f"Loaded checkpoint step {ckpt.get('global_step', '?')}")

    step_length = config.get("step_length", 400)
    n_steps = args.seq_len // (step_length + 1)  # approximate
    n_layers = model.config.num_hidden_layers

    print(f"\nSeq length: {args.seq_len} tokens (~{n_steps} steps)")
    print(f"Tree structure: B={args.branching_factor}, depth ~{math.ceil(math.log(max(1,n_steps)) / math.log(args.branching_factor))}")
    print(f"Flat GRU: {n_steps} sequential updates")
    print(f"Hierarchical: ~{args.n_recent + math.ceil(math.log(max(1,n_steps)) / math.log(args.branching_factor))} summaries decoded per step")

    # Dataset
    eval_ds = StreamingEvalDataset(
        config["dataset"], config.get("validation_split", "train"),
        tokenizer, args.seq_len, step_token_id, step_length,
    )

    results = {}

    # ---- Condition 1: Flat GRU (current approach) ----
    print(f"\n{'='*70}")
    print(f"  CONDITION 1: FLAT GRU (current)")
    print(f"{'='*70}")
    dl1 = torch.utils.data.DataLoader(eval_ds, batch_size=1)
    loss1, tok1, nb1 = eval_condition(
        model, commitment_head, pseudo_decoder, embed_layer,
        dl1, step_token_id, device, args.eval_batches,
        mode="flat_gru",
        retrieval_k=args.retrieval_k, max_bank_size=args.max_bank_size,
    )
    total_l1 = sum(loss1.values())
    total_t1 = sum(tok1.values())
    ppl_gru = math.exp(total_l1 / total_t1) if total_t1 > 0 else float('nan')
    results["flat_gru"] = {"ppl": ppl_gru, "step_loss": dict(loss1), "step_tokens": dict(tok1)}
    print(f"  Flat GRU PPL: {ppl_gru:.2f}")

    torch.cuda.empty_cache()

    # ---- Condition 2: Hierarchical Tree ----
    print(f"\n{'='*70}")
    print(f"  CONDITION 2: HIERARCHICAL TREE (B={args.branching_factor})")
    print(f"{'='*70}")
    dl2 = torch.utils.data.DataLoader(eval_ds, batch_size=1)
    loss2, tok2, nb2 = eval_condition(
        model, commitment_head, pseudo_decoder, embed_layer,
        dl2, step_token_id, device, args.eval_batches,
        mode="hierarchical",
        retrieval_k=args.retrieval_k, max_bank_size=args.max_bank_size,
        branching_factor=args.branching_factor, n_recent_l0=args.n_recent,
    )
    total_l2 = sum(loss2.values())
    total_t2 = sum(tok2.values())
    ppl_tree = math.exp(total_l2 / total_t2) if total_t2 > 0 else float('nan')
    results["hierarchical"] = {"ppl": ppl_tree, "step_loss": dict(loss2), "step_tokens": dict(tok2)}
    print(f"  Hierarchical PPL: {ppl_tree:.2f}")

    torch.cuda.empty_cache()

    # ---- Condition 3: Sliding Window (baseline) ----
    print(f"\n{'='*70}")
    print(f"  CONDITION 3: SLIDING WINDOW (800 tokens)")
    print(f"{'='*70}")
    dl3 = torch.utils.data.DataLoader(eval_ds, batch_size=1)
    loss3, tok3, nb3 = sliding_window_eval(
        model, embed_layer, tokenizer, dl3,
        step_token_id, device, args.eval_batches,
    )
    total_l3 = sum(loss3.values())
    total_t3 = sum(tok3.values())
    ppl_sw = math.exp(total_l3 / total_t3) if total_t3 > 0 else float('nan')
    results["sliding_window"] = {"ppl": ppl_sw, "step_loss": dict(loss3), "step_tokens": dict(tok3)}
    print(f"  Sliding Window PPL: {ppl_sw:.2f}")

    # ---- Summary ----
    max_step = max(
        max(loss1.keys()) if loss1 else 0,
        max(loss2.keys()) if loss2 else 0,
        max(loss3.keys()) if loss3 else 0,
    )

    print(f"\n{'='*70}")
    print(f"  HIERARCHICAL vs FLAT GRU vs SLIDING WINDOW")
    print(f"  seq_len={args.seq_len}, B={args.branching_factor}, {nb1} batches")
    print(f"{'='*70}")
    print(f"\n  {'Step':>4}  {'Flat GRU':>10}  {'Hierarchical':>12}  {'SlidWin':>10}  {'Tree-GRU':>10}  {'Tree-SW':>10}")
    print(f"  {'-'*68}")

    for step_idx in range(max_step + 1):
        ppls = {}
        for name, (loss_d, tok_d) in [("gru", (loss1, tok1)), ("tree", (loss2, tok2)), ("sw", (loss3, tok3))]:
            if step_idx in loss_d and tok_d[step_idx] > 0:
                ppls[name] = math.exp(loss_d[step_idx] / tok_d[step_idx])
            else:
                ppls[name] = float('nan')

        g = ppls.get("gru", float('nan'))
        t = ppls.get("tree", float('nan'))
        s = ppls.get("sw", float('nan'))
        tree_vs_gru = g - t if not (math.isnan(g) or math.isnan(t)) else float('nan')
        tree_vs_sw = s - t if not (math.isnan(s) or math.isnan(t)) else float('nan')

        print(f"  {step_idx:>4}  {g:>10.2f}  {t:>12.2f}  {s:>10.2f}  {tree_vs_gru:>+10.2f}  {tree_vs_sw:>+10.2f}")

    print(f"  {'-'*68}")
    tree_vs_gru_all = ppl_gru - ppl_tree
    tree_vs_sw_all = ppl_sw - ppl_tree
    print(f"  {'ALL':>4}  {ppl_gru:>10.2f}  {ppl_tree:>12.2f}  {ppl_sw:>10.2f}  {tree_vs_gru_all:>+10.2f}  {tree_vs_sw_all:>+10.2f}")

    print(f"\n  Tree-GRU > 0: hierarchical tree beats flat GRU")
    print(f"  Tree-SW  > 0: hierarchical tree beats sliding window")

    if tree_vs_gru_all > 0:
        print(f"\n  RESULT: Hierarchical tree improves over flat GRU by {tree_vs_gru_all:.2f} PPL")
    else:
        print(f"\n  RESULT: Flat GRU still better by {-tree_vs_gru_all:.2f} PPL")

    # Save
    results_dir = Path(config.get("results_dir", "./results-hierarchical"))
    results_dir.mkdir(parents=True, exist_ok=True)
    outfile = results_dir / f"hierarchical_B{args.branching_factor}_{args.seq_len}.json"
    with open(outfile, "w") as f:
        json.dump({
            "seq_len": args.seq_len,
            "branching_factor": args.branching_factor,
            "n_recent": args.n_recent,
            "n_batches": nb1,
            "flat_gru_ppl": ppl_gru,
            "hierarchical_ppl": ppl_tree,
            "sliding_window_ppl": ppl_sw,
            "tree_vs_gru": tree_vs_gru_all,
            "tree_vs_sw": tree_vs_sw_all,
        }, f, indent=2)
    print(f"\n  Saved: {outfile}")


if __name__ == "__main__":
    main()
