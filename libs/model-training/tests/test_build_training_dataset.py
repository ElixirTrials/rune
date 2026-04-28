"""Tests for ``_build_training_dataset`` and ``_build_sft_config``.

Focused on the diff-aware wiring added in Task 5:

- ``pre_code`` / ``post_code`` columns attached iff ``diff_aware_loss=True``.
- ``SFTConfig.remove_unused_columns`` is flipped to ``False`` iff
  ``diff_aware_loss=True`` so TRL does not strip the side-channel columns.
- The module is CPU-importable (INFRA-05).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pair(
    *,
    task_id: str,
    source_task_id: str,
    step_index: int,
    activation: str,
    teacher: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "activation_text": activation,
        "teacher_text": teacher,
        "metadata": {
            "source_task_id": source_task_id,
            "step_index": step_index,
            "outcome": "merged",
        },
    }


def _write_pairs_jsonl(tmp_path: Path, pairs: list[dict[str, Any]]) -> Path:
    p = tmp_path / "pairs.jsonl"
    with p.open("w") as fh:
        for rec in pairs:
            fh.write(json.dumps(rec) + "\n")
    return p


class _FakeDataset:
    """Stand-in for ``datasets.Dataset`` in CPU tests.

    Records the list passed to ``from_list`` so assertions can inspect
    column presence without importing ``datasets`` (keeps the test
    CPU-importable).
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]]) -> _FakeDataset:
        return cls(list(rows))

    def __len__(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# CPU import invariant
# ---------------------------------------------------------------------------


def test_module_is_cpu_importable() -> None:
    """trainer is importable without torch / transformers / trl / peft."""
    from model_training import trainer

    assert hasattr(trainer, "_build_training_dataset")
    assert hasattr(trainer, "_build_sft_config")


# ---------------------------------------------------------------------------
# Dataset column wiring
# ---------------------------------------------------------------------------


def test_attaches_pre_post_columns_when_diff_aware(tmp_path: Path) -> None:
    """diff_aware_loss=True → rows carry pre_code, post_code, messages."""
    from model_training.trainer import _build_training_dataset

    pairs = [
        _pair(
            task_id="t1",
            source_task_id="t1",
            step_index=0,
            activation="## Task\nAdd floats",
            teacher=(
                "## Task\nAdd floats\n\n## Implementation\n"
                "def add(a, b):\n    return a + b\n"
            ),
        ),
    ]
    path = _write_pairs_jsonl(tmp_path, pairs)

    ds = _build_training_dataset(
        dataset_cls=_FakeDataset,
        session_id=None,
        dataset_path=str(path),
        encoding_mode="single_turn",
        diff_aware_loss=True,
    )

    assert isinstance(ds, _FakeDataset)
    assert len(ds) == 1
    row = ds.rows[0]
    assert set(row.keys()) == {"messages", "pre_code", "post_code"}
    assert isinstance(row["messages"], list)
    assert isinstance(row["pre_code"], str)
    assert isinstance(row["post_code"], str)
    # Initial-commit activation_text has no "## Current Code" marker, so
    # pre_code is empty.
    assert row["pre_code"] == ""
    # post_code is the code body of the "## Implementation" section.
    assert "def add" in row["post_code"]


def test_no_pre_post_columns_when_diff_aware_false(tmp_path: Path) -> None:
    """diff_aware_loss=False → rows contain only the messages column."""
    from model_training.trainer import _build_training_dataset

    pairs = [
        _pair(
            task_id="t1",
            source_task_id="t1",
            step_index=0,
            activation="## Task\nAdd ints",
            teacher=(
                "## Task\nAdd ints\n\n## Implementation\n"
                "def add(a, b):\n    return a + b\n"
            ),
        ),
    ]
    path = _write_pairs_jsonl(tmp_path, pairs)

    ds = _build_training_dataset(
        dataset_cls=_FakeDataset,
        session_id=None,
        dataset_path=str(path),
        encoding_mode="single_turn",
        diff_aware_loss=False,
    )

    assert isinstance(ds, _FakeDataset)
    assert len(ds) == 1
    row = ds.rows[0]
    assert set(row.keys()) == {"messages"}


def test_multi_turn_diff_aware_concatenates_pre_post(tmp_path: Path) -> None:
    """Multi-turn diff-aware: pre/post are concatenated across turns."""
    from model_training.trainer import _build_training_dataset

    activation_step_1 = (
        "## Task\nAdd floats\n\n## Current Code\n"
        "def add(a, b):\n    return a + b\n\n"
        "## Review Feedback\nhandle floats"
    )
    teacher_step_1 = (
        f"{activation_step_1}\n\n## Revision\n"
        "def add(a, b):\n    return float(a) + float(b)\n"
    )
    pairs = [
        _pair(
            task_id="t1",
            source_task_id="t1",
            step_index=0,
            activation="## Task\nAdd floats",
            teacher=(
                "## Task\nAdd floats\n\n## Implementation\n"
                "def add(a, b):\n    return a + b\n"
            ),
        ),
        _pair(
            task_id="t1",
            source_task_id="t1",
            step_index=1,
            activation=activation_step_1,
            teacher=teacher_step_1,
        ),
    ]
    path = _write_pairs_jsonl(tmp_path, pairs)

    ds = _build_training_dataset(
        dataset_cls=_FakeDataset,
        session_id=None,
        dataset_path=str(path),
        encoding_mode="multi_turn",
        diff_aware_loss=True,
    )

    assert len(ds) == 1
    row = ds.rows[0]
    # Turn 1 had no ## Current Code (initial-commit), so its pre is empty;
    # turn 2's pre is the "def add(a, b): return a + b" body. Concatenated
    # with "\n\n".
    assert "def add(a, b):\n    return a + b" in row["pre_code"]
    # Two revisions concatenated with "\n\n".
    assert row["post_code"].count("def add") == 2


def test_raises_on_empty_pairs(tmp_path: Path) -> None:
    """Empty / all-skipped pairs raise ValueError regardless of diff_aware."""
    import pytest
    from model_training.trainer import _build_training_dataset

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    with pytest.raises(ValueError, match="no SFT conversations"):
        _build_training_dataset(
            dataset_cls=_FakeDataset,
            session_id=None,
            dataset_path=str(empty),
            encoding_mode="single_turn",
            diff_aware_loss=True,
        )


# ---------------------------------------------------------------------------
# SFTConfig.remove_unused_columns wiring
# ---------------------------------------------------------------------------


class _FakeSFTConfig:
    """Stand-in for ``trl.SFTConfig`` — records the kwargs it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_build_sft_config_sets_remove_unused_columns_false_when_diff_aware() -> None:
    from model_training.trainer import _build_sft_config

    cfg = _build_sft_config(
        sft_config_cls=_FakeSFTConfig,
        output_dir="/tmp/out",
        resolved_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=None,
        resolved_lr_sched="constant",
        resolved_grad_accum=16,
        report_to="none",
        diff_aware_loss=True,
        neftune_noise_alpha=None,
    )
    assert cfg.kwargs["remove_unused_columns"] is False
    assert cfg.kwargs["assistant_only_loss"] is False


def test_build_sft_config_keeps_default_remove_unused_when_diff_aware_false() -> None:
    from model_training.trainer import _build_sft_config

    cfg = _build_sft_config(
        sft_config_cls=_FakeSFTConfig,
        output_dir="/tmp/out",
        resolved_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=None,
        resolved_lr_sched="constant",
        resolved_grad_accum=16,
        report_to="none",
        diff_aware_loss=False,
        neftune_noise_alpha=None,
    )
    assert "remove_unused_columns" not in cfg.kwargs
    # assistant_only_loss is False unconditionally — TRL's
    # get_training_chat_template pre-flight (which the True branch triggers)
    # cannot patch Qwen3.5's chat template. We provide assistant_masks
    # via trajectory.compute_assistant_masks instead.
    assert cfg.kwargs["assistant_only_loss"] is False
