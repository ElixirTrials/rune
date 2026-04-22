"""Tests for mined-pair task-description augmentation.

AMENDMENT 2026-04-22: strict task_description field only; no fallbacks.
Tests verify the selector drops pairs without task_description and that the
corpus audit gate enforces MIN_RETENTION_RATIO = 0.80.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from model_training.encoder_pretrain.augment import (
    _select_task_desc,
    augment_pairs_with_task_desc,
)


def _pair(**overrides: Any) -> dict[str, Any]:
    """Build a minimal mined-pair record."""
    base: dict[str, Any] = {
        "task_id": "pr_001",
        "activation_text": "## Task\n\n## Current Code\ndef foo(): pass",
        "teacher_text": (
            "## Task\n\n## Current Code\ndef foo(): pass\n\n## Revision\ndef foo(): return 1"
        ),
        "metadata": {
            "source_task_id": "pr_001",
            "step_index": 0,
            "outcome": "merged",
            "language": None,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _select_task_desc — AMENDMENT: strict field only
# ---------------------------------------------------------------------------


def test_select_explicit_field() -> None:
    """Explicit task_description field is returned as-is."""
    pair = _pair(task_description="Fix the segfault in parser")
    result = _select_task_desc(pair)
    assert result == "Fix the segfault in parser"


def test_select_missing_returns_none() -> None:
    """Pair with no task_description key returns None (DROP)."""
    pair = _pair()  # no task_description key
    assert _select_task_desc(pair) is None


def test_select_empty_string_returns_none() -> None:
    """Pair with whitespace-only task_description returns None (DROP)."""
    pair = _pair(task_description="   ")
    assert _select_task_desc(pair) is None


def test_select_none_value_returns_none() -> None:
    """Pair with task_description=None returns None (DROP)."""
    pair = _pair(task_description=None)
    assert _select_task_desc(pair) is None


def test_select_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped from explicit field."""
    pair = _pair(task_description="  Fix the bug  ")
    result = _select_task_desc(pair)
    assert result == "Fix the bug"


# ---------------------------------------------------------------------------
# augment_pairs_with_task_desc
# ---------------------------------------------------------------------------


def test_augment_pairs_with_task_desc_adds_fields() -> None:
    """augment_pairs_with_task_desc returns rows with required schema fields."""
    pairs = [_pair(task_description="Fix the bug")]
    rows = augment_pairs_with_task_desc(pairs)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_desc"] == "Fix the bug"
    assert row["task_desc_source"] == "explicit_field"
    assert "encoder_input" in row
    assert "pre_code" in row
    assert "post_code" in row
    assert row["task_id"] == "pr_001"


def test_augment_drops_pairs_without_task_description(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pairs without task_description are dropped; count logged at WARNING."""
    pairs = [
        _pair(task_description="Fix bug A"),
        _pair(task_id="pr_002"),  # no task_description — drop
        _pair(task_id="pr_003", task_description="Fix bug C"),
    ]
    with caplog.at_level(
        logging.WARNING, logger="model_training.encoder_pretrain.augment"
    ):
        rows = augment_pairs_with_task_desc(pairs)
    assert len(rows) == 2
    assert {r["task_id"] for r in rows} == {"pr_001", "pr_003"}
    assert any("dropped 1/3" in msg for msg in caplog.messages)


def test_augment_empty_pairs() -> None:
    """Empty input returns empty list without error."""
    rows = augment_pairs_with_task_desc([])
    assert rows == []


def test_augment_logs_kept_count(caplog: pytest.LogCaptureFixture) -> None:
    """A summary log line at INFO level reports kept count."""
    pairs = [
        _pair(task_description="desc a"),
        _pair(task_id="pr_002", task_description="desc b"),
    ]
    with caplog.at_level(
        logging.INFO, logger="model_training.encoder_pretrain.augment"
    ):
        augment_pairs_with_task_desc(pairs)
    assert any("kept 2" in msg for msg in caplog.messages)


def test_augment_task_desc_source_is_always_explicit_field() -> None:
    """task_desc_source is always 'explicit_field'."""
    pairs = [_pair(task_description="Some task")]
    rows = augment_pairs_with_task_desc(pairs)
    assert rows[0]["task_desc_source"] == "explicit_field"


# ---------------------------------------------------------------------------
# augment_corpus
# ---------------------------------------------------------------------------


def test_augment_corpus_missing_dir_raises(tmp_path: Path) -> None:
    """augment_corpus raises FileNotFoundError when pairs_dir does not exist."""
    from model_training.encoder_pretrain.augment import augment_corpus

    missing = tmp_path / "nonexistent_pairs"
    with pytest.raises(FileNotFoundError, match="pairs_dir"):
        augment_corpus(pairs_dir=missing, output_dir=tmp_path / "out")


def test_augment_corpus_writes_jsonl(tmp_path: Path) -> None:
    """augment_corpus writes augmented JSONL for each input JSONL file."""
    import json

    from model_training.d2l_data import save_jsonl
    from model_training.encoder_pretrain.augment import augment_corpus

    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    out_dir = tmp_path / "pairs_augmented"

    pair = _pair(task_description="Fix bug", task_id="pr_001")
    save_jsonl([pair], pairs_dir / "test_repo.jsonl")

    augment_corpus(pairs_dir=pairs_dir, output_dir=out_dir)

    out_file = out_dir / "test_repo.jsonl"
    assert out_file.exists()
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["task_desc_source"] == "explicit_field"


def test_augment_corpus_fails_below_retention_threshold(tmp_path: Path) -> None:
    """augment_corpus raises RuntimeError when retention < 80%."""
    from model_training.d2l_data import save_jsonl
    from model_training.encoder_pretrain.augment import augment_corpus

    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    out_dir = tmp_path / "pairs_augmented"

    # 1 with desc, 4 without — 20% retention (below 80% threshold)
    mixed = [_pair(task_description="has desc")] + [
        _pair(task_id=f"pr_{i}") for i in range(4)
    ]
    save_jsonl(mixed, pairs_dir / "repo.jsonl")
    with pytest.raises(RuntimeError, match="task_description"):
        augment_corpus(pairs_dir=pairs_dir, output_dir=out_dir)


def test_augment_corpus_passes_at_exactly_80_percent(tmp_path: Path) -> None:
    """augment_corpus succeeds when exactly 80% of pairs have task_description."""
    from model_training.d2l_data import save_jsonl
    from model_training.encoder_pretrain.augment import augment_corpus

    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    out_dir = tmp_path / "pairs_augmented"

    # 8 with desc, 2 without — exactly 80% retention
    good = [_pair(task_id=f"pr_{i}", task_description=f"Task {i}") for i in range(8)]
    bad = [_pair(task_id=f"pr_bad_{i}") for i in range(2)]
    save_jsonl(good + bad, pairs_dir / "repo.jsonl")

    # Should not raise
    augment_corpus(pairs_dir=pairs_dir, output_dir=out_dir)
    assert (out_dir / "repo.jsonl").exists()
