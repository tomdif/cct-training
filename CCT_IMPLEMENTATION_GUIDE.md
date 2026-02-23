# CCT Real-Model Training: Implementation Guide for Claude Code

## Overview

This document is a step-by-step implementation guide for training a real transformer model under Commitment-Constrained Training (CCT) and validating the Working Memory architecture (F19). The goal is to empirically prove or disprove whether commitment summaries are sufficient statistics for downstream reasoning in real transformers, enabling 100-1000× KV cache compression.

**The experiment is tiered. Stop after any tier if results are negative. Scale up if positive.**

- **Tier 1**: 160M params, 1× A100, ~$62, proves architecture works
- **Tier 2**: 1.3B params, 1-2× H100, ~$315, produces publishable result
- **Tier 3**: 7B params, 8× H100, ~$6K, competitive benchmark
- **Tier 4**: 13-70B params, 32-64× H100, ~$67K, frontier demonstration

---

## PHASE 0: Local Setup & Code Structure

**Do this phase entirely locally before renting any GPUs.**

### 0.1 Project Structure

```
cct-training/
├── README.md
├── pyproject.toml
├── configs/
│   ├── tier1_160m.yaml
│   ├── tier2_1b.yaml
│   ├── tier3_7b.yaml
│   └── tier4_70b.yaml
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cct_attention.py        # Core: modified attention with step masking
│   │   ├── commitment_head.py      # Bottleneck projection to d_summary
│   │   ├── summary_buffer.py       # F19 working memory buffer
│   │   └── gradient_isolation.py   # Stop-gradient at step boundaries
│   ├── training/
│   │   ├── __init__.py
│   │   ├── curriculum.py           # 4-phase commitment probability scheduler
│   │   ├── loss.py                 # Irreversibility-aware loss function
│   │   ├── data_pipeline.py        # Step boundary annotation
│   │   └── trainer.py              # Main training loop with CCT
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── working_memory.py       # F19 inference with summary buffer
│   │   └── governance.py           # Commitment hash chain (optional)
│   └── evaluation/
│       ├── __init__.py
│       ├── sufficiency.py          # THE critical test: summary sufficiency
│       ├── compression.py          # Memory measurement
│       ├── benchmarks.py           # Standard LM benchmarks
│       ├── passkey_retrieval.py    # Long-range recall test
│       └── comparison.py           # Head-to-head vs baselines
├── scripts/
│   ├── train.py                    # Entry point
│   ├── evaluate.py                 # Evaluation entry point
│   ├── cloud_setup.sh              # GPU cloud provisioning
│   └── run_all_tiers.sh            # Sequential tier execution
└── results/
    └── .gitkeep
```

### 0.2 Dependencies

```toml
# pyproject.toml
[project]
name = "cct-training"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "datasets>=2.16.0",
    "accelerate>=0.25.0",
    "wandb",
    "pyyaml",
    "lm-eval>=0.4.0",        # For standard benchmarks
    "numpy",
    "tqdm",
    "safetensors",
]

[project.optional-dependencies]
multi-gpu = ["deepspeed>=0.12.0"]
```

---

## PHASE 1: Core CCT Module Implementation

**This is the hardest engineering step. Get this right before touching GPUs.**

### 1.1 Step Boundary Detection (`data_pipeline.py`)

The model needs to know where reasoning step boundaries are. For training, we annotate data with explicit boundary tokens.

```python
"""
Step Boundary Annotation Strategy:
  
For reasoning data (math, logic, code):
  - Each inferential transition is a step boundary
  - e.g., in chain-of-thought: each "Step N:" or "\n\n" between reasoning blocks
  
For general text:
  - Sentence boundaries (period + space + capital letter)
  - Paragraph boundaries (\n\n)
  - Or fixed token-count windows (e.g., every 50 tokens)
  
Implementation:
  - Insert a special <STEP> token at each boundary
  - The tokenizer must be extended with this token
  - Step boundaries create the partitioning for attention masking
"""

# CRITICAL DESIGN DECISION: Step length
# - Too short (5-10 tokens): model can't reason within a step
# - Too long (200+ tokens): compression ratio suffers
# - Sweet spot for initial experiments: 32-64 tokens per step
# - For Tier 1, use fixed 50-token windows to simplify
# - For Tier 2+, use semantic boundary detection

STEP_TOKEN = "<STEP>"
STEP_TOKEN_ID = None  # Set after tokenizer extension

def annotate_step_boundaries(token_ids: list[int], mode: str = "fixed", step_length: int = 50) -> list[int]:
    """Insert STEP_TOKEN_ID at step boundaries.
    
    Args:
        token_ids: Original token sequence
        mode: "fixed" (every N tokens) or "semantic" (at sentence/paragraph boundaries)
        step_length: For fixed mode, tokens per step
    
    Returns:
        Token sequence with STEP_TOKEN_ID inserted at boundaries
    """
    # For Tier 1: use fixed mode
    # For Tier 2+: implement semantic mode using period/newline detection
    ...
```

### 1.2 CCT Attention Mask (`cct_attention.py`)

**This is the core mechanism.** The attention mask must block token-level attention across step boundaries while allowing attention to commitment summaries.

```python
"""
CCT Attention Mask Construction

Standard causal mask:
  Token i can attend to tokens 0..i (lower triangular)

CCT mask:
  Token i in step K can attend to:
    1. All tokens in step K (within-step attention — normal)
    2. Commitment summary vectors h_1..h_{K-1} (cross-step attention — summaries only)
    3. CANNOT attend to individual tokens from steps 1..K-1

Implementation approach:
  - Detect step boundaries in the input sequence
  - Build a block-diagonal mask for within-step attention
  - Add summary token positions as attendable for all tokens in later steps
  
The commitment summary for step K is computed from step K's tokens and 
projected through the commitment head to d_summary dimensions. This summary
is then injected as a "virtual token" that subsequent steps can attend to.

Two implementation strategies:

STRATEGY A (Recommended for Tier 1-2): Summary Injection
  - After processing step K, compute summary vector
  - Inject summary as a special token into the sequence for step K+1's attention
  - The attention mask allows step K+1 tokens to attend to these injected summary tokens
  - Simple to implement, compatible with HuggingFace models

STRATEGY B (For Tier 3+): Custom Attention Kernel  
  - Modify the attention computation directly
  - Maintain separate summary buffer
  - Cross-step attention computed over summary buffer instead of KV cache
  - More efficient but requires custom CUDA kernels
"""

import torch

def build_cct_attention_mask(
    seq_len: int,
    step_boundary_positions: list[int],  # indices where STEP tokens appear
    num_prior_summaries: int,            # number of summary tokens prepended
    device: torch.device,
) -> torch.Tensor:
    """Build the CCT attention mask.
    
    The input sequence is structured as:
    [summary_1, summary_2, ..., summary_{K-1}, token_1, token_2, ..., token_L]
    
    Where summary_* are the commitment summaries from prior steps,
    and token_* are the current step's tokens.
    
    Returns a mask of shape (seq_len + num_prior_summaries, seq_len + num_prior_summaries)
    where True = CAN attend, False = CANNOT attend.
    """
    total_len = seq_len + num_prior_summaries
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    
    # 1. Summary tokens can attend to all prior summaries (causal over summaries)
    for i in range(num_prior_summaries):
        mask[i, :i+1] = True  # summary i attends to summaries 0..i
    
    # 2. Current step's tokens: causal attention within the step + all summaries
    for i in range(num_prior_summaries, total_len):
        # Can attend to all summary tokens
        mask[i, :num_prior_summaries] = True
        # Can attend to all prior tokens within current step (causal)
        mask[i, num_prior_summaries:i+1] = True
    
    # 3. If there are multiple steps in the current sequence (during training),
    #    tokens in step K+1 CANNOT attend to raw tokens in step K
    #    They CAN only attend to step K's summary
    #    (This is the key constraint that forces sufficiency)
    
    # For multi-step training sequences, enforce block-diagonal within-step attention
    # plus summary-only cross-step attention
    if len(step_boundary_positions) > 0:
        # Build step assignments
        step_of_token = []
        current_step = 0
        for pos in range(seq_len):
            if pos in step_boundary_positions:
                current_step += 1
            step_of_token.append(current_step)
        
        # Mask out cross-step token attention
        for i in range(seq_len):
            for j in range(seq_len):
                if step_of_token[j] < step_of_token[i]:
                    # j is in an earlier step than i
                    # Block this attention (only summaries allowed)
                    mask[i + num_prior_summaries, j + num_prior_summaries] = False
    
    return mask
```

### 1.3 Commitment Head (`commitment_head.py`)

```python
"""
Commitment Head: Projects step-final hidden states to d_summary.

This is the information bottleneck that forces the model to compress
all cross-step-relevant information into a fixed-size vector.

Architecture:
  hidden_state (d_model) → Linear(d_model, d_bottleneck) → LayerNorm → 
  Tanh → Linear(d_bottleneck, d_summary) → L2 normalize

The bottleneck dimension should be MUCH smaller than d_model:
  - d_model=768 (160M model):  d_summary=16-32
  - d_model=2048 (1.3B model): d_summary=32-64
  - d_model=4096 (7B model):   d_summary=64-128
  
The L2 normalization ensures summaries live on a hypersphere,
which stabilizes attention over summaries and makes the
sufficiency measurement well-defined.
"""

import torch
import torch.nn as nn

class CommitmentHead(nn.Module):
    def __init__(self, d_model: int, d_summary: int, d_bottleneck: int = None):
        super().__init__()
        if d_bottleneck is None:
            d_bottleneck = d_summary * 2
        
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
            nn.Tanh(),
            nn.Linear(d_bottleneck, d_summary),
        )
    
    def forward(self, step_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_hidden_states: (batch, step_length, d_model) 
                                Hidden states from the last transformer layer for one step.
        Returns:
            summary: (batch, d_summary) — the commitment summary vector
        """
        # Use the last token's hidden state as input (like [CLS] pooling)
        # Alternative: mean pool over step tokens
        last_token = step_hidden_states[:, -1, :]  # (batch, d_model)
        summary = self.projection(last_token)        # (batch, d_summary)
        summary = nn.functional.normalize(summary, dim=-1)  # L2 normalize
        return summary
```

### 1.4 Gradient Isolation (`gradient_isolation.py`)

```python
"""
Gradient Isolation Barrier

The forward pass proceeds normally: step K+1 reads commitment summaries from steps 1..K.
The backward pass BLOCKS gradients from step K+1's loss through the summaries back into step K.

This means step K's parameters are optimized ONLY by step K's own loss.
Step K cannot "cheat" by encoding information that only helps downstream steps.

Implementation: torch.detach() on the commitment summary before passing to next step.
"""

def isolate_gradient(summary: torch.Tensor) -> torch.Tensor:
    """Detach summary from computation graph.
    
    The summary value is preserved for forward pass (next step can read it),
    but no gradient flows back through it during backward pass.
    """
    return summary.detach()

# CRITICAL: This must be applied AFTER the commitment head computes the summary
# and BEFORE the summary is used as input context for the next step.
# 
# In the training loop:
#   summary = commitment_head(step_k_hidden_states)
#   summary_for_next_step = isolate_gradient(summary)  # <-- THIS LINE
#   # Now use summary_for_next_step as context for step K+1
```

### 1.5 Curriculum Scheduler (`curriculum.py`)

```python
"""
4-Phase Commitment Probability Curriculum

Controls the probability that commitment is enforced at each step boundary.
When NOT enforced: standard autoregressive training (no mask, no gradient isolation).
When enforced: full CCT constraints apply.

The model doesn't know which steps will be committed at generation time,
matching the inference-time uncertainty.
"""

class CommitmentCurriculum:
    def __init__(self, total_steps: int):
        self.total = total_steps
        # Phase boundaries as fractions of total training
        self.phase_boundaries = [0.10, 0.50, 0.80, 1.00]
    
    def get_commitment_probability(self, current_step: int) -> float:
        """Returns p_commit for current training step."""
        progress = current_step / self.total
        
        if progress < self.phase_boundaries[0]:
            # Phase 1: Awareness — no commitment enforcement
            return 0.0
        
        elif progress < self.phase_boundaries[1]:
            # Phase 2: Stochastic — p_commit linearly 0.1 → 0.5
            phase_progress = (progress - 0.10) / 0.40
            return 0.1 + phase_progress * 0.4
        
        elif progress < self.phase_boundaries[2]:
            # Phase 3: Majority — p_commit linearly 0.5 → 0.9
            phase_progress = (progress - 0.50) / 0.30
            return 0.5 + phase_progress * 0.4
        
        else:
            # Phase 4: Full Commitment — p_commit = 1.0
            return 1.0
    
    def get_loss_weights(self, current_step: int) -> dict:
        """Returns loss component weights for current training step.
        
        alpha: standard autoregressive loss weight (decreases)
        beta: per-step validity loss weight (increases)
        gamma: conclusion-from-premises loss weight (increases)
        delta: sufficiency loss weight (increases)
        """
        progress = current_step / self.total
        
        # Smooth transition from standard training to commitment training
        commitment_weight = min(1.0, max(0.0, (progress - 0.1) / 0.7))
        
        return {
            'alpha': 1.0 - 0.5 * commitment_weight,   # Standard LM loss: 1.0 → 0.5
            'beta':  0.5 * commitment_weight,           # Per-step validity: 0.0 → 0.5
            'gamma': 0.3 * commitment_weight,           # Conclusion from premises: 0.0 → 0.3
            'delta': 0.2 * commitment_weight,           # Sufficiency: 0.0 → 0.2
        }
```

### 1.6 Loss Function (`loss.py`)

```python
"""
Irreversibility-Aware Loss Function

Four components, weighted by curriculum:

1. alpha * L_standard:   Standard next-token prediction (cross-entropy)
2. beta  * L_validity:   Per-step validity — each step correct given ONLY prior summaries
3. gamma * L_conclusion: Conclusion from premises — final answer from summaries alone
4. delta * L_sufficiency: Explicit sufficiency — can a probe recover step info from summary?

The key insight: L_validity and L_conclusion are what FORCE sufficiency.
If the model can only use summaries (not raw tokens) to produce correct 
next-step predictions, it MUST put everything needed into the summaries.
"""

import torch
import torch.nn.functional as F

def compute_cct_loss(
    # Standard autoregressive loss (computed normally)
    lm_logits: torch.Tensor,           # (batch, seq_len, vocab_size)
    labels: torch.Tensor,               # (batch, seq_len)
    
    # Per-step validity: recompute predictions using ONLY summaries as context
    validity_logits: torch.Tensor,       # (batch, step_len, vocab_size) 
    validity_labels: torch.Tensor,       # (batch, step_len)
    
    # Conclusion from premises: predict final answer from summaries only
    conclusion_logits: torch.Tensor,     # (batch, answer_len, vocab_size)
    conclusion_labels: torch.Tensor,     # (batch, answer_len)
    
    # Sufficiency probe: linear probe reconstructing step content from summary
    summary: torch.Tensor,              # (batch, d_summary)
    step_content_target: torch.Tensor,  # (batch, d_model) — mean hidden state of step
    sufficiency_probe: torch.nn.Linear, # d_summary → d_model
    
    # Weights from curriculum
    weights: dict,
) -> dict:
    """Compute the full CCT loss."""
    
    # 1. Standard autoregressive loss
    L_standard = F.cross_entropy(
        lm_logits.view(-1, lm_logits.size(-1)), 
        labels.view(-1), 
        ignore_index=-100
    )
    
    # 2. Per-step validity loss
    L_validity = F.cross_entropy(
        validity_logits.view(-1, validity_logits.size(-1)),
        validity_labels.view(-1),
        ignore_index=-100
    )
    
    # 3. Conclusion from premises loss
    L_conclusion = F.cross_entropy(
        conclusion_logits.view(-1, conclusion_logits.size(-1)),
        conclusion_labels.view(-1),
        ignore_index=-100
    )
    
    # 4. Sufficiency loss — how well can a linear probe reconstruct from summary?
    reconstructed = sufficiency_probe(summary)
    L_sufficiency = F.mse_loss(reconstructed, step_content_target.detach())
    
    # Combine
    total_loss = (
        weights['alpha'] * L_standard +
        weights['beta']  * L_validity +
        weights['gamma'] * L_conclusion +
        weights['delta'] * L_sufficiency
    )
    
    return {
        'total': total_loss,
        'standard': L_standard.item(),
        'validity': L_validity.item(),
        'conclusion': L_conclusion.item(),
        'sufficiency': L_sufficiency.item(),
    }
```

### 1.7 Training Loop (`trainer.py`)

```python
"""
Main CCT Training Loop

For each batch:
  1. Tokenize and annotate step boundaries
  2. Get commitment probability from curriculum
  3. For each step in the sequence:
     a. Run the transformer on step tokens (with prior summaries as context)
     b. Compute commitment summary via commitment head
     c. If committed: apply gradient isolation, store summary
     d. Compute per-step validity loss using summary-only context
  4. Compute conclusion-from-premises loss
  5. Compute total loss and backpropagate
  6. Log all metrics to wandb

Key implementation detail:
  The model sees the ENTIRE sequence in one forward pass (for efficiency),
  but the attention mask enforces the step structure.
  
  During the forward pass:
  - All tokens attend within their step (block-diagonal for current step)
  - All tokens attend to commitment summary tokens from prior steps
  - No token attends to raw tokens from prior steps
  
  This is implemented by modifying the attention mask, NOT by running
  separate forward passes per step.
"""

# SIMPLIFIED TRAINING LOOP (Tier 1 version):
#
# for batch in dataloader:
#     input_ids, step_boundaries = annotate_step_boundaries(batch)
#     
#     # Build CCT attention mask
#     mask = build_cct_attention_mask(input_ids, step_boundaries, p_commit)
#     
#     # Forward pass with modified mask
#     outputs = model(input_ids, attention_mask=mask)
#     hidden_states = outputs.last_hidden_state
#     
#     # Compute commitment summaries at each step boundary
#     summaries = []
#     for boundary in step_boundaries:
#         step_hidden = hidden_states[:, boundary-step_len:boundary, :]
#         summary = commitment_head(step_hidden)
#         summary = isolate_gradient(summary)
#         summaries.append(summary)
#     
#     # Compute losses
#     loss_dict = compute_cct_loss(...)
#     
#     # Backprop
#     loss_dict['total'].backward()
#     optimizer.step()
#     
#     # Log
#     wandb.log(loss_dict)
```

---

## PHASE 2: Tier 1 — Proof of Concept (160M parameters)

### 2.1 Cloud Setup

```bash
#!/bin/bash
# cloud_setup.sh — Run this on a fresh GPU VM

# Option A: VAST.ai (cheapest, ~$0.75/hr for A100)
# 1. Go to vast.ai, create account
# 2. Search for A100 80GB instances
# 3. Select one with PyTorch 2.1+ image
# 4. SSH in, then run this script

# Option B: RunPod ($1.29/hr A100, $2.49/hr H100)
# Option C: Lambda Labs ($1.29/hr A100, $2.49/hr H100)
# Option D: Hyperbolic ($1.49/hr H100 — cheapest H100)

# Install dependencies
pip install torch transformers datasets accelerate wandb pyyaml lm-eval safetensors tqdm

# Clone your repo (push to private GitHub first)
git clone https://github.com/YOUR_USERNAME/cct-training.git
cd cct-training

# Login to wandb for experiment tracking
wandb login YOUR_API_KEY

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

### 2.2 Tier 1 Config

```yaml
# configs/tier1_160m.yaml

# Model
base_model: "EleutherAI/pythia-160m"  # or "EleutherAI/pythia-160m-deduped"
d_model: 768
n_layers: 12
n_heads: 12

# CCT Architecture
d_summary: 16          # Commitment summary dimension (768 → 16 = 48× compression per step)
d_bottleneck: 32       # Intermediate bottleneck in commitment head
step_length: 50        # Fixed token count per step (Tier 1 simplification)
step_boundary_mode: "fixed"  # "fixed" for Tier 1, "semantic" for Tier 2+

# Training
total_steps: 50000     # ~25M tokens at batch_size=8, seq_len=512
batch_size: 8
seq_len: 512           # = 10 steps of 50 tokens each (plus boundaries)
learning_rate: 5e-5    # Fine-tuning LR (not training from scratch)
warmup_steps: 1000
weight_decay: 0.01
max_grad_norm: 1.0

# Curriculum
curriculum:
  phase1_end: 0.10     # Steps 0-5000: awareness
  phase2_end: 0.50     # Steps 5000-25000: stochastic commitment
  phase3_end: 0.80     # Steps 25000-40000: majority commitment
  phase4_end: 1.00     # Steps 40000-50000: full commitment

# Data
dataset: "cerebras/SlimPajama-627B"  # Or "allenai/c4"
dataset_streaming: true

# Evaluation (run every 5000 steps)
eval_interval: 5000
eval_tasks:
  - sufficiency_probe    # LINEAR PROBE: can we recover step info from summary?
  - compression_ratio    # MEMORY: actual KV cache vs summary buffer size
  - lm_perplexity        # QUALITY: perplexity on held-out data
  - passkey_retrieval    # RECALL: can model find a passkey across step boundaries?

# Comparison baseline
train_baseline: true    # Also train same model WITHOUT CCT for comparison
```

### 2.3 Training Commands

```bash
# Step 1: Train baseline (standard fine-tuning, no CCT)
python scripts/train.py \
  --config configs/tier1_160m.yaml \
  --mode baseline \
  --wandb_project cct-tier1 \
  --wandb_name baseline-160m

# Step 2: Train CCT model
python scripts/train.py \
  --config configs/tier1_160m.yaml \
  --mode cct \
  --wandb_project cct-tier1 \
  --wandb_name cct-160m

# Step 3: Evaluate both
python scripts/evaluate.py \
  --models baseline-160m cct-160m \
  --tasks sufficiency compression perplexity passkey \
  --output results/tier1_results.json
```

### 2.4 What to Measure (Tier 1 Success Criteria)

**CRITICAL MEASUREMENT: Sufficiency Probe**

```python
"""
The sufficiency probe answers the ONE question that matters:

"After CCT training, does the commitment summary contain enough information
 for downstream steps to reason correctly?"

Method:
  1. Take the CCT-trained model
  2. Run it on held-out data with step boundaries
  3. At each step boundary, extract:
     a. The commitment summary (d_summary vector)
     b. The full hidden state of the step (d_model vector, mean-pooled)
  4. Train a LINEAR probe: summary → hidden state
  5. Measure reconstruction error

If the linear probe can reconstruct the full hidden state from the summary
with low error, the summary IS a sufficient statistic.

Comparison: run the same probe on the BASELINE model's outputs.
The baseline model wasn't trained to compress into summaries,
so its probe error should be MUCH higher.
"""

# SUCCESS CRITERIA for Tier 1:
TIER1_PASS = {
    'sufficiency_probe_r2':    0.85,    # R² > 0.85 (summary captures 85%+ of step info)
    'perplexity_delta':        0.10,    # CCT perplexity < 10% worse than baseline
    'compression_ratio':       50,      # At least 50× memory compression
    'passkey_accuracy_1k':     0.90,    # 90%+ passkey retrieval at 1K tokens
    'passkey_accuracy_4k':     0.70,    # 70%+ passkey retrieval at 4K tokens
}

# If ALL criteria pass → proceed to Tier 2
# If sufficiency probe fails → the thesis is wrong, STOP
# If perplexity delta too high → curriculum needs tuning, iterate
# If compression ratio low → increase step length or reduce d_summary
```

**Memory Compression Measurement:**

```python
"""
Measure actual memory usage:

Standard model at context length L:
  KV cache = n_layers × 2 × L × d_model × sizeof(float16)
  
CCT model at same context length L with step_length S:
  Summary buffer = (L / S) × d_summary × sizeof(float32)
  Local KV cache = n_layers × 2 × S × d_model × sizeof(float16) (bounded!)
  
Compression ratio = standard KV cache / (summary buffer + local KV cache)

For Tier 1 (768 d_model, 12 layers, 50-token steps, 16 d_summary):
  Standard at 4K tokens: 12 × 2 × 4096 × 768 × 2 bytes = 150 MB
  CCT summary buffer:    80 × 16 × 4 bytes = 5 KB
  CCT local cache:       12 × 2 × 50 × 768 × 2 bytes = 1.8 MB
  Compression: 150 MB / 1.8 MB ≈ 83×
  
  Standard at 32K tokens: 1.2 GB
  CCT summary buffer:     640 × 16 × 4 bytes = 40 KB  
  CCT local cache:        1.8 MB (SAME — bounded by step length!)
  Compression: 1.2 GB / 1.8 MB ≈ 667×
"""
```

### 2.5 Passkey Retrieval Test

```python
"""
Passkey Retrieval: The gold-standard test for long-range information retention.

Insert a random passkey (e.g., "The passkey is 83719") at position P in a long context.
Fill the rest with irrelevant padding text.
Ask the model to recall the passkey at the end.

For CCT: the passkey must survive the commitment summary bottleneck.
If the step containing the passkey produces a summary that encodes the passkey,
and a later step can extract it from that summary, sufficiency holds.

Test at context lengths: 512, 1K, 2K, 4K, 8K, 16K, 32K
Compare: CCT model vs baseline vs Mamba (if available)
"""
```

---

## PHASE 3: Tier 2 — 1.3B Validation (Publishable Result)

### 3.1 Tier 2 Config Changes

```yaml
# configs/tier2_1b.yaml — changes from Tier 1

base_model: "EleutherAI/pythia-1.4b"  # Or "meta-llama/Llama-3.2-1B"
d_model: 2048
n_layers: 24
n_heads: 16

d_summary: 32           # Larger model → slightly larger summary
d_bottleneck: 64
step_length: 50          # Same step length → higher compression
step_boundary_mode: "semantic"  # Upgrade to semantic boundaries

total_steps: 100000      # ~100M tokens
batch_size: 4            # Reduced for memory
seq_len: 1024            # = 20 steps per sequence
learning_rate: 2e-5

# GPU
gpus: 1                  # 1× H100 80GB sufficient for 1.4B
mixed_precision: "bf16"

# Additional evaluations
eval_tasks:
  - sufficiency_probe
  - compression_ratio
  - lm_perplexity
  - passkey_retrieval
  - hellaswag           # Standard benchmark
  - arc_easy            # Standard benchmark
  - winogrande          # Standard benchmark
  - mmlu_5shot          # Standard benchmark (if time permits)
```

### 3.2 Tier 2 Success Criteria

```python
TIER2_PASS = {
    'sufficiency_probe_r2':    0.90,    # Higher bar at 1B+ scale
    'perplexity_delta':        0.05,    # CCT within 5% of baseline perplexity
    'compression_ratio_8k':    200,     # 200×+ at 8K context
    'compression_ratio_32k':   1000,    # 1000×+ at 32K context
    'passkey_accuracy_8k':     0.85,    # 85%+ passkey at 8K
    'passkey_accuracy_32k':    0.60,    # 60%+ passkey at 32K
    'hellaswag_delta':         0.05,    # Within 5% of baseline on HellaSwag
    'governance_overhead':     0.02,    # <2% compute overhead for commitment hashing
}

# If ALL pass → this is publishable. Write the paper. File non-provisionals.
# If compression passes but quality degrades → iterate on curriculum/loss weights
# If passkey fails at long range → check summary dimensionality
```

### 3.3 Head-to-Head Comparison Table (what the paper reports)

```
| Metric                  | Pythia-1.4B | CCT-1.4B | Mamba-1.4B | RWKV-7 1.5B |
|------------------------|-------------|----------|------------|--------------|
| HellaSwag              |     XX.X    |   XX.X   |    XX.X    |     XX.X     |
| ARC-Easy               |     XX.X    |   XX.X   |    XX.X    |     XX.X     |
| Perplexity (C4)        |     XX.X    |   XX.X   |    XX.X    |     XX.X     |
| Passkey@8K             |     XX%     |   XX%    |    XX%     |     XX%      |
| Passkey@32K            |     XX%     |   XX%    |    XX%     |     XX%      |
| KV Memory @ 32K        |   XX MB     |  XX MB   |  XX MB     |    XX MB     |
| Compression Ratio      |     1×      |  XXX×    |    ∞       |      ∞       |
| Governance Capable     |     No      |   Yes    |    No      |     No       |
| Information Loss Bound |   N/A       |  Zero    | Unbounded  |  Unbounded   |
```

---

## PHASE 4: Tier 3 — 7B Competitive Demo

**Only proceed if Tier 2 passes.**

### 4.1 Two Paths at 7B

```yaml
# PATH A: CCT Fine-tuning (cheaper, faster, uses F18 Claim 10)
base_model: "meta-llama/Llama-3.1-8B"
mode: "finetune"           # Apply CCT as fine-tuning, not from-scratch
total_steps: 50000         # Fine-tuning requires fewer steps
gpus: 8                    # 8× H100

# PATH B: CCT from scratch (stronger result but more expensive)
base_model: null           # Train from scratch
architecture: "llama"      # Same architecture as Llama
total_steps: 200000        # ~200B tokens
gpus: 8
```

### 4.2 Tier 3 Evaluation Suite

```bash
# Full lm-evaluation-harness benchmark suite
lm_eval --model hf \
  --model_args pretrained=./cct-7b \
  --tasks hellaswag,arc_easy,arc_challenge,winogrande,mmlu,truthfulqa_mc2,gsm8k \
  --device cuda \
  --batch_size 8

# Long-context evaluation (RULER benchmark)
python scripts/evaluate.py \
  --model ./cct-7b \
  --tasks ruler_niah ruler_vt ruler_cwe ruler_fwe \
  --context_lengths 4096 8192 16384 32768 65536 131072

# Memory profiling
python scripts/evaluate.py \
  --model ./cct-7b \
  --tasks memory_profile \
  --context_lengths 8192 32768 131072 524288 1048576

# Governance overhead
python scripts/evaluate.py \
  --model ./cct-7b \
  --tasks governance_overhead \
  --tiers 1 2 3
```

---

## PHASE 5: Evaluation & Reporting

### 5.1 The Five Numbers That Matter

After each tier, report these five numbers:

```python
"""
1. SUFFICIENCY R²
   How much of the step's information survives the commitment bottleneck?
   Method: Linear probe from d_summary → d_model
   Target: > 0.90 at Tier 2+
   
2. COMPRESSION RATIO AT 32K CONTEXT
   How much smaller is the summary buffer than the standard KV cache?
   Method: Actual byte measurement
   Target: > 1000× at Tier 2+
   
3. PERPLEXITY DELTA vs BASELINE
   How much quality do we lose from CCT constraints?
   Method: Perplexity on held-out C4
   Target: < 5% worse than same-architecture baseline
   
4. PASSKEY ACCURACY AT 32K
   Can the model retrieve specific information across many step boundaries?
   Method: Standard passkey retrieval test
   Target: > 80% at Tier 2+ (Mamba/RWKV typically fail here)
   
5. GOVERNANCE OVERHEAD
   What's the compute cost of commitment hashing?
   Method: Time per token with vs without governance
   Target: < 2% overhead
"""
```

### 5.2 Results JSON Format

```json
{
  "tier": 2,
  "model": "cct-pythia-1.4b",
  "baseline": "pythia-1.4b",
  "training": {
    "total_steps": 100000,
    "tokens_seen": "100M",
    "gpu_hours": 150,
    "cost_usd": 315,
    "final_loss": {
      "total": 0.0,
      "standard": 0.0,
      "validity": 0.0,
      "conclusion": 0.0,
      "sufficiency": 0.0
    }
  },
  "results": {
    "sufficiency_probe_r2": 0.0,
    "compression_ratio": {
      "4k": 0,
      "8k": 0,
      "32k": 0,
      "128k": 0
    },
    "perplexity": {
      "cct": 0.0,
      "baseline": 0.0,
      "delta_pct": 0.0
    },
    "passkey_accuracy": {
      "1k": 0.0,
      "4k": 0.0,
      "8k": 0.0,
      "32k": 0.0
    },
    "benchmarks": {
      "hellaswag": 0.0,
      "arc_easy": 0.0,
      "winogrande": 0.0
    },
    "governance_overhead_pct": 0.0
  },
  "conclusion": "PASS/FAIL/ITERATE",
  "next_action": ""
}
```

---

## Implementation Order for Claude Code

**Follow this exact sequence:**

### Sprint 1: Core Architecture (local, no GPU needed)
1. Set up project structure per Section 0.1
2. Implement `data_pipeline.py` — step boundary annotation (fixed mode)
3. Implement `commitment_head.py` — bottleneck projection
4. Implement `gradient_isolation.py` — simple detach wrapper
5. Implement `cct_attention.py` — build_cct_attention_mask function
6. **Unit test**: Create a tiny 2-layer transformer, verify the attention mask blocks cross-step token attention, verify gradient isolation prevents cross-step gradient flow
7. Implement `curriculum.py` — 4-phase scheduler
8. Implement `loss.py` — all four loss components

### Sprint 2: Training Loop (local, no GPU needed for code)
9. Implement `trainer.py` — main training loop integrating all components
10. Implement `sufficiency.py` — linear probe evaluation
11. Implement `compression.py` — memory measurement
12. Implement `passkey_retrieval.py` — passkey test
13. **Integration test**: Run 100 steps on CPU with a tiny model (2 layers, d_model=64, d_summary=4) to verify the full pipeline works end-to-end. Loss should decrease. Commitment probability should follow curriculum.

### Sprint 3: Tier 1 Training (GPU — ~$62)
14. Provision A100 80GB instance on VAST.ai or RunPod
15. Upload code, install dependencies
16. Train baseline Pythia-160M (standard fine-tuning, no CCT) — ~12 hours
17. Train CCT Pythia-160M — ~24 hours
18. Run full evaluation suite
19. Compare results against Tier 1 success criteria
20. **Decision gate**: If sufficiency R² > 0.85 and perplexity delta < 10% → proceed. Otherwise iterate.

### Sprint 4: Tier 2 Training (GPU — ~$315)
21. Provision H100 80GB instance
22. Switch to Pythia-1.4B or Llama-3.2-1B
23. Implement semantic step boundary detection
24. Train baseline — ~50 hours
25. Train CCT — ~100 hours
26. Run full evaluation suite including standard benchmarks
27. Generate comparison table for paper
28. **Decision gate**: This IS the publishable result.

### Sprint 5: Tier 3 (conditional — ~$6K)
29. Only if Tier 2 passes all criteria
30. Set up 8× H100 cluster with DeepSpeed/FSDP
31. CCT fine-tune Llama-3.1-8B
32. Full benchmark suite + RULER long-context evaluation
33. Memory profiling at 128K and 1M context lengths
34. Generate final comparison: CCT-8B vs Llama-8B vs Mamba-2.8B vs RWKV-7 2.9B

---

## Key Pitfalls to Avoid

### Pitfall 1: Attention Mask Compatibility
HuggingFace transformer models expect specific attention mask formats. The CCT mask is non-standard (not just causal). You may need to:
- Use `model.forward(attention_mask=custom_mask)` 
- Some models accept 2D masks, others need 4D (batch, heads, seq, seq)
- Flash Attention v2 may not support arbitrary masks — fall back to standard attention for initial implementation
- **Test the mask shape and dtype on the specific model before training**

### Pitfall 2: Summary Injection Strategy
For Strategy A (summary injection), the summaries need to be treated as "virtual tokens" in the sequence. Two approaches:
- **Approach 1**: Prepend summary embeddings to the input before the transformer. This requires modifying the positional encoding.
- **Approach 2**: Replace the STEP token's hidden state with the computed summary after the first pass. This is simpler but requires two forward passes per sequence.
- **For Tier 1, use Approach 2** (simpler implementation, perf doesn't matter yet)

### Pitfall 3: Gradient Isolation Timing
The `detach()` must happen AFTER the commitment head computes the summary and BEFORE the summary is used as context. If you detach too early (before the commitment head), the commitment head doesn't train. If you detach too late (after it's used as context), gradient flows through and you lose the CCT property.

### Pitfall 4: Curriculum Phase Transitions
The transition from Phase 1 (no commitment) to Phase 2 (stochastic commitment) can cause a loss spike as the model suddenly encounters masked attention. This is expected and normal. If the spike doesn't recover within 1000 steps, the learning rate is too low.

### Pitfall 5: Step Length Sensitivity
If steps are too short, the summary must compress very little information and the test is trivially easy. If steps are too long, the bottleneck is too severe and quality degrades. Start with 50 tokens/step and sweep [25, 50, 100] in Tier 1 to find the sweet spot.

### Pitfall 6: Sufficiency Probe Overfitting
The linear probe must be trained on a HELD-OUT set of step/summary pairs, not the training data. If the probe overfits, R² will be artificially high. Use a separate validation set and early stopping.

---

## Budget Summary

| Tier | GPU | Hours | Rate | Total | Proves |
|------|-----|-------|------|-------|--------|
| 0 | None (local) | N/A | $0 | $0 | Code works |
| 1 | 1× A100 | 48 | $1.29 | $62 | Architecture works on real transformers |
| 2 | 1× H100 | 150 | $2.10 | $315 | Publishable result at 1B+ scale |
| 3 | 8× H100 | 350 | $2.10/gpu | $5,880 | Competitive 7B benchmark |
| 4 | 32× H100 | 1000 | $2.10/gpu | $67,200 | Frontier demonstration |

**Cumulative cost to publishable result: $377**
**Cumulative cost to competitive 7B: ~$6,300**

---

## After Results: Next Steps

### If Tier 2 passes:
1. Write arXiv preprint: "Commitment-Constrained Training: Provably Sufficient Summaries Enable 1000× KV Cache Compression"
2. File non-provisional patents on F18 + F19 with empirical data in specification
3. Open-source the code (establishes priority, attracts community)
4. Submit to ICML/NeurIPS
5. Begin Tier 3

### If Tier 2 fails on sufficiency but passes on quality:
- The bottleneck is too aggressive. Increase d_summary.
- Or the training curriculum needs more Phase 3/4 steps.
- This is an engineering problem, not a thesis-breaking result.

### If Tier 2 fails on quality:
- CCT constraints hurt model quality more than expected at this scale.
- Check if Phase 1 (awareness) is long enough for the model to learn basics.
- Try starting from a better pre-trained model (Llama-3.2-1B) instead of training from scratch.
- If quality is 15%+ worse even after iteration, the thesis needs revision.

### If Tier 2 fails on sufficiency AND quality:
- The thesis is wrong. Commitment summaries in real transformers are not sufficient statistics.
- The patents still have governance value (F16/F17 don't depend on sufficiency).
- But the working memory (F19) and compression claims don't hold.
- This is a $315 lesson. Better than $50K in patent prosecution fees.
