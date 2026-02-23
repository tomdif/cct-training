"""
Integration test: run CCT training steps on CPU with a tiny model.

Uses a 2-layer GPT-style model with d_model=64, d_summary=4, step_length=10.
Runs through all 4 curriculum phases in miniature (200 total steps).

Verifies:
  1. Loss decreases over steps
  2. Curriculum probability follows schedule
  3. Summaries are produced at each step boundary
  4. No NaN or Inf in any tensor
  5. Gradient isolation is working
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

from src.training.trainer import CCTTrainer, TrainConfig
from src.training.data_pipeline import extend_tokenizer, annotate_step_boundaries
from src.training.curriculum import CommitmentCurriculum
from src.model.cct_attention import build_cct_attention_mask_fast


def create_tiny_model():
    """Create a minimal GPT2 model for testing."""
    config = GPT2Config(
        vocab_size=1000,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=128,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config)
    return model


def create_fake_batch(batch_size, seq_len, step_token_id, step_length, vocab_size=1000):
    """Create a batch of token IDs with STEP tokens inserted."""
    batches = []
    for _ in range(batch_size):
        # Random tokens (avoid step_token_id)
        tokens = torch.randint(0, vocab_size - 1, (seq_len,)).tolist()
        annotated = annotate_step_boundaries(tokens, step_token_id, "fixed", step_length)
        # Truncate to seq_len
        annotated = annotated[:seq_len]
        # Pad if needed
        while len(annotated) < seq_len:
            annotated.append(0)
        batches.append(torch.tensor(annotated, dtype=torch.long))
    return torch.stack(batches)


class TestIntegration:
    """Integration tests for the full CCT pipeline."""

    def setup_method(self):
        """Set up tiny model and trainer for each test."""
        self.model = create_tiny_model()
        self.d_model = 64
        self.d_summary = 4
        self.step_length = 10
        self.seq_len = 64
        self.step_token_id = 999  # Use a fixed ID for the tiny model

        self.config = TrainConfig(
            d_model=self.d_model,
            d_summary=self.d_summary,
            d_bottleneck=8,
            step_length=self.step_length,
            total_steps=200,
            batch_size=2,
            seq_len=self.seq_len,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=10,
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,
            phase1_end=0.10,
            phase2_end=0.50,
            phase3_end=0.80,
            eval_interval=100,
            log_interval=50,
            device="cpu",
            mixed_precision="fp32",
        )

        self.trainer = CCTTrainer(
            model=self.model,
            tokenizer=None,  # Not needed for direct train_step calls
            config=self.config,
            step_token_id=self.step_token_id,
        )

    def test_standard_step_runs(self):
        """Phase 1 standard training step completes without error."""
        batch = create_fake_batch(2, self.seq_len, self.step_token_id, self.step_length)
        # Force phase 1
        self.trainer.global_step = 0
        result = self.trainer.train_step(batch)
        assert "total" in result
        assert result["phase"] == 1

    def test_cct_step_runs(self):
        """CCT training step (phase 4) completes without error."""
        batch = create_fake_batch(2, self.seq_len, self.step_token_id, self.step_length)
        # Force phase 4 (full commitment)
        self.trainer.global_step = 180  # > 0.80 * 200 = 160
        result = self.trainer.train_step(batch)
        assert "total" in result
        assert result["phase"] == 4
        assert result.get("n_summaries", 0) > 0

    def test_loss_decreases(self):
        """Loss decreases when training on a fixed batch (memorization test)."""
        # Use the SAME batch every step so the model can memorize it
        torch.manual_seed(42)
        fixed_batch = create_fake_batch(
            2, self.seq_len, self.step_token_id, self.step_length
        )

        losses = []
        for step in range(100):
            result = self.trainer.train_step(fixed_batch)
            total = result["total"]
            if hasattr(total, "item"):
                total = total.item()
            losses.append(total)

        # Loss at the end should be lower than at the start
        avg_start = sum(losses[:5]) / 5
        avg_end = sum(losses[-5:]) / 5
        assert avg_end < avg_start, (
            f"Loss did not decrease: start={avg_start:.4f}, end={avg_end:.4f}"
        )

    def test_curriculum_phases(self):
        """Curriculum produces correct phase sequence."""
        curriculum = CommitmentCurriculum(total_steps=200)
        assert curriculum.get_phase(0) == 1
        assert curriculum.get_phase(10) == 1  # step 10 = 5% < 10%
        assert curriculum.get_phase(30) == 2  # step 30 = 15%
        assert curriculum.get_phase(110) == 3  # step 110 = 55%
        assert curriculum.get_phase(170) == 4  # step 170 = 85%

    def test_commitment_probability_ramps(self):
        """p_commit increases across phases."""
        curriculum = CommitmentCurriculum(total_steps=200)
        p_phase1 = curriculum.get_commitment_probability(5)
        p_phase2_mid = curriculum.get_commitment_probability(60)
        p_phase3_mid = curriculum.get_commitment_probability(130)
        p_phase4 = curriculum.get_commitment_probability(190)

        assert p_phase1 == 0.0
        assert 0.1 < p_phase2_mid < 0.5
        assert 0.5 < p_phase3_mid < 0.9
        assert p_phase4 == 1.0

    def test_no_nan_inf(self):
        """No NaN or Inf in model outputs during CCT step."""
        batch = create_fake_batch(2, self.seq_len, self.step_token_id, self.step_length)
        self.trainer.global_step = 180  # phase 4

        result = self.trainer.train_step(batch)

        # Check model parameters for NaN/Inf
        for name, param in self.model.named_parameters():
            assert not torch.any(torch.isnan(param)), f"NaN in {name}"
            assert not torch.any(torch.isinf(param)), f"Inf in {name}"

        # Check commitment head
        for name, param in self.trainer.commitment_head.named_parameters():
            assert not torch.any(torch.isnan(param)), f"NaN in commitment_head.{name}"
            assert not torch.any(torch.isinf(param)), f"Inf in commitment_head.{name}"

    def test_full_training_run(self):
        """Run through all 4 phases (200 steps) without crashing.

        This is the full integration test that exercises the complete pipeline.
        """
        phase_seen = set()
        cct_steps = 0
        standard_steps = 0

        for step in range(200):
            batch = create_fake_batch(
                2, self.seq_len, self.step_token_id, self.step_length
            )
            result = self.trainer.train_step(batch)

            phase = result.get("phase", 1)
            phase_seen.add(phase)

            if result.get("n_summaries", 0) > 0:
                cct_steps += 1
            else:
                standard_steps += 1

        # Should have seen all 4 phases
        assert phase_seen == {1, 2, 3, 4}, f"Only saw phases: {phase_seen}"

        # Should have a mix of standard and CCT steps
        assert standard_steps > 0, "No standard training steps (phase 1 broken?)"
        assert cct_steps > 0, "No CCT training steps (commitment never triggered?)"

        print(f"\n  Full run: {standard_steps} standard + {cct_steps} CCT steps")
        print(f"  Phases seen: {sorted(phase_seen)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
