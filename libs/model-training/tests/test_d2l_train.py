"""Unit tests for D2LTrainConfig and _compute_kl_ce_loss.

Tests are CPU-only and pure-tensor — no GPU, no model loading.
"""

from __future__ import annotations

import json

import pytest
import torch

from model_training.d2l_train import D2LTrainConfig, _compute_kl_ce_loss


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
        """config.model_dump() returns dict with all fields, JSON-serializable via json.dumps."""
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

    def _make_config(self, alpha: float = 0.5, temperature: float = 2.0) -> D2LTrainConfig:
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
        assert metrics["kl_loss"] < 1e-5, f"Expected KL ~0 for equal logits, got {metrics['kl_loss']}"

    def test_kl_positive_when_different(self) -> None:
        """KL divergence is positive when student and teacher logits differ."""
        config = self._make_config(alpha=0.5)
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)
        _, metrics = _compute_kl_ce_loss(student, teacher, answer_start=0, config=config)
        assert metrics["kl_loss"] > 0, f"Expected KL > 0 for different logits, got {metrics['kl_loss']}"

    def test_ce_loss_valid_shape(self) -> None:
        """CE loss on answer span produces a finite scalar tensor."""
        config = self._make_config(alpha=0.5)
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)
        total_loss, metrics = _compute_kl_ce_loss(student, teacher, answer_start=0, config=config)
        assert total_loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(total_loss), "Loss should be finite"
        assert torch.isfinite(torch.tensor(metrics["ce_loss"])), "CE loss should be finite"

    def test_blended_loss_respects_alpha(self) -> None:
        """alpha=1.0 gives total==KL, alpha=0.0 gives total==CE (within tolerance)."""
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)

        config_kl = self._make_config(alpha=1.0)
        _, metrics_kl = _compute_kl_ce_loss(student, teacher, answer_start=0, config=config_kl)
        assert abs(metrics_kl["total_loss"] - metrics_kl["kl_loss"]) < 1e-5, (
            f"alpha=1.0: total should equal KL. total={metrics_kl['total_loss']}, kl={metrics_kl['kl_loss']}"
        )

        config_ce = self._make_config(alpha=0.0)
        _, metrics_ce = _compute_kl_ce_loss(student, teacher, answer_start=0, config=config_ce)
        assert abs(metrics_ce["total_loss"] - metrics_ce["ce_loss"]) < 1e-5, (
            f"alpha=0.0: total should equal CE. total={metrics_ce['total_loss']}, ce={metrics_ce['ce_loss']}"
        )

    def test_answer_start_masking(self) -> None:
        """With answer_start=3 on seq_len=5, loss only considers last 2 positions.

        Verify by comparing loss with full-sequence (answer_start=0) vs masked (answer_start=3).
        Different logit slices → different loss values.
        """
        config = self._make_config(alpha=0.5)
        student = torch.randn(1, 5, 10)
        teacher = torch.randn(1, 5, 10)

        _, metrics_full = _compute_kl_ce_loss(student, teacher, answer_start=0, config=config)
        _, metrics_masked = _compute_kl_ce_loss(student, teacher, answer_start=3, config=config)

        # Loss values must differ when slicing a different span (with very high probability for random logits)
        assert metrics_full["total_loss"] != metrics_masked["total_loss"], (
            "Masking to answer_start=3 should produce different loss than full sequence"
        )
