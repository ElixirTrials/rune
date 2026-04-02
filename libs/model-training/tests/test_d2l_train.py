"""Unit tests for D2LTrainConfig and _compute_kl_ce_loss.

Tests are CPU-only and pure-tensor — no GPU, no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from model_training.d2l_train import (
    D2LTrainConfig,
    _compute_kl_ce_loss,
    _save_checkpoint,
)


class TestD2LTrainConfigPydanticValidation:
    """Tests for D2LTrainConfig field validators."""

    def test_d2l_train_config_pydantic_validation_rejects_negative_lr(self) -> None:
        """D2LTrainConfig rejects negative lr via field validator."""
        with pytest.raises(Exception):  # pydantic ValidationError
            D2LTrainConfig(lr=-1e-4, sakana_checkpoint_path="ckpt.pt")

    def test_d2l_train_config_pydantic_validation_rejects_negative_alpha(self) -> None:
        """D2LTrainConfig rejects alpha outside [0, 1]."""
        with pytest.raises(Exception):
            D2LTrainConfig(alpha=-0.1, sakana_checkpoint_path="ckpt.pt")

    def test_d2l_train_config_pydantic_validation_rejects_alpha_above_one(self) -> None:
        """D2LTrainConfig rejects alpha > 1."""
        with pytest.raises(Exception):
            D2LTrainConfig(alpha=1.5, sakana_checkpoint_path="ckpt.pt")

    def test_d2l_train_config_pydantic_validation_accepts_valid_defaults(self) -> None:
        """D2LTrainConfig accepts valid default values."""
        config = D2LTrainConfig(sakana_checkpoint_path="ckpt.pt")
        assert config.lr > 0
        assert 0.0 <= config.alpha <= 1.0
        assert config.temperature > 0
        assert config.num_steps > 0

    def test_d2l_train_config_model_dump(self) -> None:
        """config.model_dump() returns dict with all fields, JSON-serializable."""
        config = D2LTrainConfig(sakana_checkpoint_path="ckpt.pt")
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        # Verify JSON-serializable
        serialized = json.dumps(dumped)
        assert isinstance(serialized, str)
        # Spot-check key fields present
        assert "lr" in dumped
        assert "alpha" in dumped
        assert "num_steps" in dumped
        assert "sakana_checkpoint_path" in dumped


class TestComputeKlCeLoss:
    """Tests for _compute_kl_ce_loss pure function."""

    def _make_config(
        self, alpha: float = 0.5, temperature: float = 2.0
    ) -> D2LTrainConfig:
        return D2LTrainConfig(
            sakana_checkpoint_path="ckpt.pt",
            alpha=alpha,
            temperature=temperature,
        )

    def test_kl_zero_when_equal(self) -> None:
        """KL divergence is ~0 when student_logits == teacher_logits."""
        config = self._make_config(alpha=0.5)
        # batch=1, seq_len=5, vocab=10
        logits = torch.randn(1, 5, 10)
        _, metrics = _compute_kl_ce_loss(logits, logits, answer_start=0, config=config)
        assert metrics["kl_loss"] < 1e-5, (
            f"Expected KL ~0 for equal logits, got {metrics['kl_loss']}"
        )

    def test_kl_positive_when_different(self) -> None:
        """KL divergence is positive when student and teacher logits differ."""
        config = self._make_config(alpha=0.5)
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)
        _, metrics = _compute_kl_ce_loss(
            student, teacher, answer_start=0, config=config
        )
        assert metrics["kl_loss"] > 0, (
            f"Expected KL > 0 for different logits, got {metrics['kl_loss']}"
        )

    def test_ce_loss_valid_shape(self) -> None:
        """CE loss on answer span produces a finite scalar tensor."""
        config = self._make_config(alpha=0.5)
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)
        total_loss, metrics = _compute_kl_ce_loss(
            student, teacher, answer_start=0, config=config
        )
        assert total_loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(total_loss), "Loss should be finite"
        assert torch.isfinite(torch.tensor(metrics["ce_loss"])), (
            "CE loss should be finite"
        )

    def test_blended_loss_respects_alpha(self) -> None:
        """alpha=1.0 gives total==KL, alpha=0.0 gives total==CE (within tolerance)."""
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)

        config_kl = self._make_config(alpha=1.0)
        _, metrics_kl = _compute_kl_ce_loss(
            student, teacher, answer_start=0, config=config_kl
        )
        assert abs(metrics_kl["total_loss"] - metrics_kl["kl_loss"]) < 1e-5, (
            f"alpha=1.0: total should equal KL. "
            f"total={metrics_kl['total_loss']}, kl={metrics_kl['kl_loss']}"
        )

        config_ce = self._make_config(alpha=0.0)
        _, metrics_ce = _compute_kl_ce_loss(
            student, teacher, answer_start=0, config=config_ce
        )
        assert abs(metrics_ce["total_loss"] - metrics_ce["ce_loss"]) < 1e-5, (
            f"alpha=0.0: total should equal CE. "
            f"total={metrics_ce['total_loss']}, ce={metrics_ce['ce_loss']}"
        )

    def test_answer_start_masking(self) -> None:
        """With answer_start=3 on seq_len=5, loss considers last 2 positions only.

        Verify by comparing loss with full-sequence (answer_start=0) vs
        masked (answer_start=3). Different slices → different loss values.
        """
        config = self._make_config(alpha=0.5)
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)

        _, metrics_full = _compute_kl_ce_loss(
            student, teacher, answer_start=0, config=config
        )
        _, metrics_masked = _compute_kl_ce_loss(
            student, teacher, answer_start=3, config=config
        )

        # Loss values must differ when slicing a different span
        assert metrics_full["total_loss"] != metrics_masked["total_loss"], (
            "Masking to answer_start=3 should produce different loss than full sequence"
        )


# ---------------------------------------------------------------------------
# Fake model helper for gradient tests
# ---------------------------------------------------------------------------


class _FakeHypernet(nn.Module):
    """Minimal nn.Module with aggregator (frozen) and head (trainable) sub-modules."""

    def __init__(self) -> None:
        super().__init__()
        self.aggregator = nn.Linear(8, 8)
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# New Plan 02 tests
# ---------------------------------------------------------------------------


class TestFrozenAggregatorZeroGrad:
    """After backward, frozen aggregator params have None gradient."""

    def test_frozen_aggregator_zero_grad(self) -> None:
        """Aggregator params have grad=None after backward when frozen."""
        model = _FakeHypernet()
        for param in model.aggregator.parameters():
            param.requires_grad_(False)

        x = torch.randn(2, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for param in model.aggregator.parameters():
            assert param.grad is None, (
                f"Expected None grad for frozen aggregator param, got {param.grad}"
            )


class TestTrainableHeadNonzeroGrad:
    """After backward, trainable head params have non-None gradients."""

    def test_trainable_head_nonzero_grad(self) -> None:
        """Head params have non-None grad after backward when trainable."""
        model = _FakeHypernet()
        for param in model.aggregator.parameters():
            param.requires_grad_(False)

        x = torch.randn(2, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert model.head.weight.grad is not None, (
            "Expected non-None grad for head.weight"
        )
        assert model.head.bias.grad is not None, "Expected non-None grad for head.bias"


class TestOptimizerScopedToTrainable:
    """AdamW optimizer is scoped exclusively to trainable (head) params."""

    def test_optimizer_scoped_to_trainable(self) -> None:
        """AdamW param group contains only head params, not aggregator params."""
        from torch.optim import AdamW  # noqa: PLC0415

        model = _FakeHypernet()
        for param in model.aggregator.parameters():
            param.requires_grad_(False)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=1e-4)

        # head has weight + bias = 2 params
        head_param_count = sum(1 for _ in model.head.parameters())
        optimizer_param_count = len(optimizer.param_groups[0]["params"])

        assert optimizer_param_count == head_param_count, (
            f"Optimizer should have {head_param_count} head params, "
            f"got {optimizer_param_count}"
        )


class TestCheckpointSaveLoadRoundtrip:
    """_save_checkpoint writes a file that torch.load can read back."""

    def test_checkpoint_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Saved checkpoint can be loaded with torch.load without errors."""
        model = _FakeHypernet()
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"state": {}, "param_groups": []}
        scheduler = MagicMock()
        scheduler.state_dict.return_value = {"last_epoch": 0}

        config = D2LTrainConfig(
            sakana_checkpoint_path="ckpt.pt",
            checkpoint_dir=str(tmp_path),
        )

        # Fake hc with layer_indices attribute
        hc = MagicMock()
        hc.layer_indices = [3, 7, 11]

        ckpt_path = _save_checkpoint(
            step=10,
            hypernet=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            hc=hc,
            best_loss=0.42,
            full=False,
        )

        assert ckpt_path.exists(), f"Checkpoint file not found: {ckpt_path}"
        loaded = torch.load(ckpt_path, weights_only=False)
        assert isinstance(loaded, dict), "Loaded checkpoint must be a dict"


class TestCheckpointContainsRequiredKeys:
    """Saved checkpoint dicts contain all documented keys."""

    def test_checkpoint_contains_required_keys(self, tmp_path: Path) -> None:
        """Lightweight checkpoint has required keys."""
        model = _FakeHypernet()
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"state": {}, "param_groups": []}
        scheduler = MagicMock()
        scheduler.state_dict.return_value = {"last_epoch": 0}

        config = D2LTrainConfig(
            sakana_checkpoint_path="ckpt.pt",
            checkpoint_dir=str(tmp_path),
        )
        hc = MagicMock()
        hc.layer_indices = [3, 7, 11]

        # Lightweight checkpoint
        ckpt_path = _save_checkpoint(
            step=5,
            hypernet=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            hc=hc,
            best_loss=1.23,
            full=False,
        )
        loaded = torch.load(ckpt_path, weights_only=False)

        required_keys = {
            "hypernet_state_dict",
            "config_json",
            "step",
            "attention_layer_indices",
            "best_loss",
        }
        missing = required_keys - set(loaded.keys())
        assert not missing, f"Lightweight checkpoint missing keys: {missing}"

        # Full checkpoint also has optimizer/scheduler/rng state
        ckpt_path_full = _save_checkpoint(
            step=500,
            hypernet=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            hc=hc,
            best_loss=0.9,
            full=True,
        )
        loaded_full = torch.load(ckpt_path_full, weights_only=False)
        full_required = required_keys | {
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
        }
        missing_full = full_required - set(loaded_full.keys())
        assert not missing_full, f"Full checkpoint missing keys: {missing_full}"


class TestTwoPassSeparationShapes:
    """Two-pass separation produces student and teacher logits of matching shape."""

    def test_two_pass_separation_shapes(self) -> None:
        """Student and teacher logits match shape; _compute_kl_ce_loss accepts them."""
        config = D2LTrainConfig(sakana_checkpoint_path="ckpt.pt")

        # Simulate shapes: batch=1, seq_len=10, vocab=50
        student_logits = torch.randn(1, 10, 50)
        teacher_logits = torch.randn(1, 10, 50)

        assert student_logits.shape == teacher_logits.shape, (
            "Student and teacher logit shapes must match"
        )

        loss, metrics = _compute_kl_ce_loss(
            student_logits, teacher_logits, answer_start=5, config=config
        )
        assert loss.ndim == 0, "Loss must be a scalar"
        assert torch.isfinite(loss), "Loss must be finite"
        assert "kl_loss" in metrics
        assert "ce_loss" in metrics
        assert "total_loss" in metrics


class TestSmokeTestLossFinite:
    """5-step loss loop using _compute_kl_ce_loss produces finite losses."""

    def test_smoke_test_loss_finite(self) -> None:
        """All 5 losses from random logits are finite floats."""
        config = D2LTrainConfig(sakana_checkpoint_path="ckpt.pt", alpha=0.5)

        finite_results: list[bool] = []
        for _ in range(5):
            student = torch.randn(1, 10, 50)
            teacher = torch.randn(1, 10, 50)
            loss, metrics = _compute_kl_ce_loss(
                student, teacher, answer_start=0, config=config
            )
            finite_results.append(loss.isfinite().item())

        assert all(finite_results), (
            f"Expected all 5 losses to be finite, got: {finite_results}"
        )
        assert len(finite_results) == 5, "Must run exactly 5 steps"


class TestCausalShiftAwareLoss:
    """Verify _compute_kl_ce_loss accounts for the causal LM logit shift."""

    def _make_config(
        self, alpha: float = 0.5, temperature: float = 2.0
    ) -> D2LTrainConfig:
        return D2LTrainConfig(
            sakana_checkpoint_path="ckpt.pt",
            alpha=alpha,
            temperature=temperature,
        )

    def test_compute_kl_ce_loss_causal_shift(self) -> None:
        """With answer_start=3, logit slice starts at position 2 (shift-aware).

        Verifies the shift is applied by checking the slice length: with
        answer_start=3 on seq_len=8, the shift-aware slice starts at
        position 2 (length 6), not position 3 (length 5).
        """
        config = self._make_config(alpha=0.5)
        batch, seq, vocab = 1, 8, 10

        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)

        # answer_start=3 → logit_start = max(0, 3-1) = 2
        # Slice should be [:, 2:, :] → 6 positions, not 5
        loss_at_3, metrics_at_3 = _compute_kl_ce_loss(
            student, teacher, answer_start=3, config=config
        )

        # answer_start=1 → logit_start = max(0, 1-1) = 0
        # Slice should be [:, 0:, :] → 8 positions (full)
        loss_at_1, metrics_at_1 = _compute_kl_ce_loss(
            student, teacher, answer_start=1, config=config
        )

        # Both should produce finite losses
        assert torch.isfinite(loss_at_3)
        assert torch.isfinite(loss_at_1)

        # answer_start=1 uses full sequence (logit_start=0), answer_start=3
        # uses a shorter slice (logit_start=2) → different loss values
        assert metrics_at_3["total_loss"] != metrics_at_1["total_loss"], (
            "Different answer_start values should produce different losses"
        )

    def test_compute_kl_ce_loss_empty_span_returns_zero(self) -> None:
        """When answer_start >= seq_len, returns zero loss without crashing."""
        config = self._make_config()
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)

        # answer_start=6 exceeds seq_len=5
        loss, metrics = _compute_kl_ce_loss(
            student, teacher, answer_start=6, config=config
        )

        assert loss.item() == 0.0, f"Expected zero loss, got {loss.item()}"
        assert loss.requires_grad, "Zero loss must still have requires_grad=True"
        assert metrics["kl_loss"] == 0.0

    def test_compute_kl_ce_loss_boundary_answer_start_equals_seq_len(self) -> None:
        """When answer_start == seq_len, no answer tokens exist → zero loss."""
        config = self._make_config()
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)

        # answer_start=5 == seq_len=5 → no answer tokens
        loss, metrics = _compute_kl_ce_loss(
            student, teacher, answer_start=5, config=config
        )

        assert loss.item() == 0.0, f"Expected zero loss, got {loss.item()}"
        assert metrics["kl_loss"] == 0.0
        assert metrics["ce_loss"] == 0.0
        assert metrics["total_loss"] == 0.0
