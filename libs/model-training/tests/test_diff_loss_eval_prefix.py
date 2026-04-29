"""CPU test: DiffAwareSFTTrainer.log() prefixes accumulated metrics by context.

Train-context dicts (no ``eval_loss`` key) get ``train/*`` (existing behaviour).
Eval-context dicts (``eval_loss`` present) get ``eval/*`` so the
OptunaScreeningCallback can read them.

Uses mock.patch on the SFTTrainer parent to avoid GPU initialization while
still exercising the real log() method through Python's MRO.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _make_trainer_instance() -> tuple:
    """Create a DiffAwareSFTTrainer instance with mocked __init__ and parent log."""
    from model_training.diff_loss import DiffAwareSFTTrainer

    with patch.object(DiffAwareSFTTrainer, "__init__", lambda self: None):
        inst = DiffAwareSFTTrainer()
    inst._diff_metric_sums = {}
    inst._diff_metric_count = 0
    return inst, DiffAwareSFTTrainer


def test_log_uses_train_prefix_when_no_eval_loss() -> None:
    inst, cls = _make_trainer_instance()
    inst._diff_metric_sums = {"token_accuracy": 0.84, "entropy": 0.6}
    inst._diff_metric_count = 1

    captured: dict[str, float] = {}

    def fake_parent_log(self, logs, start_time=None):
        captured.update(logs)

    parent_cls = cls.__mro__[1]  # SFTTrainer or whatever is next in MRO
    with patch.object(parent_cls, "log", fake_parent_log):
        inst.log({"loss": 1.2})

    assert "train/token_accuracy" in captured
    assert captured["train/token_accuracy"] == pytest.approx(0.84)
    assert "eval/token_accuracy" not in captured


def test_log_uses_eval_prefix_when_eval_loss_present() -> None:
    inst, cls = _make_trainer_instance()
    inst._diff_metric_sums = {"token_accuracy": 0.91, "entropy": 0.55}
    inst._diff_metric_count = 1

    captured: dict[str, float] = {}

    def fake_parent_log(self, logs, start_time=None):
        captured.update(logs)

    parent_cls = cls.__mro__[1]
    with patch.object(parent_cls, "log", fake_parent_log):
        inst.log({"eval_loss": 1.05})

    assert "eval/token_accuracy" in captured
    assert captured["eval/token_accuracy"] == pytest.approx(0.91)
    assert "eval/entropy" in captured
    assert "train/token_accuracy" not in captured


def test_log_resets_accumulator_after_flush() -> None:
    inst, cls = _make_trainer_instance()
    inst._diff_metric_sums = {"token_accuracy": 0.5}
    inst._diff_metric_count = 1

    parent_cls = cls.__mro__[1]
    with patch.object(parent_cls, "log", lambda self, logs, start_time=None: None):
        inst.log({"loss": 1.0})

    assert inst._diff_metric_sums == {}
    assert inst._diff_metric_count == 0
