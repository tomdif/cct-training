# CLAUDE.md — Instructions for Claude Code

## What This Project Is

CCT (Commitment-Constrained Training) is a novel training method for transformer language models. It trains the model to produce fixed-size "commitment summaries" at reasoning step boundaries that are **sufficient statistics** for all downstream reasoning. This enables:
- Replacing the KV cache with a tiny summary buffer (2,400× memory compression)
- Provably lossless information retention across step boundaries
- Cryptographic governance proofs as a free byproduct

This repo implements CCT training on real transformer models and validates the working memory architecture.

## How to Build This

**Read `CCT_IMPLEMENTATION_GUIDE.md` FIRST. It contains the full technical specification, code architecture, implementation order, pitfalls, and success criteria.**

Follow the Implementation Order exactly:
1. Sprint 1: Core Architecture (local)
2. Sprint 2: Training Loop (local)  
3. Sprint 3: Tier 1 Training (GPU)
4. Sprint 4: Tier 2 Training (GPU)

## Critical Implementation Details

### The Five Core Mechanisms
1. **Attention Mask Modification**: Block token-level attention across step boundaries. Tokens in step K+1 CANNOT attend to raw tokens in step K. They CAN attend to step K's commitment summary.
2. **Commitment Head**: A bottleneck MLP that projects step-final hidden states from d_model down to d_summary (e.g., 768 → 16). This is the information bottleneck.
3. **Gradient Isolation**: `summary.detach()` after the commitment head, before passing to next step. Step K's parameters learn only from step K's loss, not downstream.
4. **4-Phase Curriculum**: Commitment probability ramps from 0% to 100% over training. Phase 1 (0-10%): no commitment. Phase 2 (10-50%): p_commit 0.1→0.5. Phase 3 (50-80%): p_commit 0.5→0.9. Phase 4 (80-100%): p_commit 1.0.
5. **Irreversibility-Aware Loss**: Four components: standard LM loss (decreasing weight), per-step validity loss, conclusion-from-premises loss, sufficiency probe loss.

### The One Measurement That Matters
**Sufficiency Probe R²**: Train a linear probe to reconstruct step hidden states from commitment summaries. If R² > 0.85, the summaries are sufficient. This is the make-or-break metric.

### Known Pitfalls
- HuggingFace models may not accept arbitrary 2D attention masks — check the model's forward() signature
- Flash Attention may not support non-causal custom masks — disable it for initial implementation
- The `detach()` for gradient isolation must happen AFTER commitment head, BEFORE summary injection
- Phase 1→2 curriculum transition will cause a loss spike — this is expected
- Sufficiency probe must be evaluated on HELD-OUT data, not training data

## Architecture

```
src/model/cct_attention.py      ← Builds the block-diagonal + summary attention mask
src/model/commitment_head.py    ← Bottleneck projection d_model → d_summary
src/model/summary_buffer.py     ← Ordered buffer of commitment summaries (F19)
src/model/gradient_isolation.py ← Stop-gradient wrapper
src/training/curriculum.py      ← 4-phase commitment probability scheduler
src/training/loss.py            ← Irreversibility-aware loss (4 components)
src/training/data_pipeline.py   ← Step boundary annotation
src/training/trainer.py         ← Main training loop
src/evaluation/sufficiency.py   ← THE critical linear probe test
src/evaluation/compression.py   ← Memory measurement
src/evaluation/passkey_retrieval.py ← Long-range recall test
src/evaluation/benchmarks.py    ← Standard LM evals via lm-eval-harness
```

## Testing Strategy

### Unit Tests (Sprint 1)
- Verify attention mask shape and values: cross-step token attention = 0, within-step = causal, summary attention = 1
- Verify gradient isolation: after detach, grad of step K params w.r.t. step K+1 loss = 0
- Verify commitment head output shape: (batch, d_summary) with L2 norm = 1

### Integration Test (Sprint 2)
- Run 100 training steps on CPU with tiny model (2 layers, d_model=64, d_summary=4, step_length=10)
- Verify: loss decreases, curriculum probability follows schedule, summaries are produced, no NaN/Inf

### Tier 1 Validation (Sprint 3)
- Train Pythia-160M with and without CCT
- Pass criteria: sufficiency R² > 0.85, perplexity delta < 10%, compression > 50×

## Quick Reference: Tier 1 Config

```yaml
base_model: "EleutherAI/pythia-160m"
d_summary: 16
step_length: 50
total_steps: 50000
batch_size: 8
seq_len: 512
learning_rate: 5e-5
gpu: 1x A100 80GB
estimated_cost: $62
```

## Dependencies

```
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0
wandb
lm-eval>=0.4.0
```

## What Success Looks Like

Tier 1 output: A JSON file showing that a 160M parameter transformer trained under CCT produces commitment summaries where:
- A linear probe achieves R² > 0.85 reconstructing step info from summaries
- Perplexity is within 10% of the same model trained normally
- Memory compression exceeds 50× at 4K context

If you see those numbers, the thesis holds. Scale to Tier 2.
