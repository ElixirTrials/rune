"""Tests for d2l_external — external code-review dataset ingestion."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from model_training.d2l_external import (
    codereview_row_to_pair,
    ingest_codereview_to_pairs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ROW: dict[str, Any] = {
    "before_code": "def foo():\n    pass",
    "after_code": "def foo():\n    return 42",
    "reviewer_comment": "The function should return a value instead of passing",
    "repo_name": "test/repo",
    "pr_number": 123,
    "file_path": "src/main.py",
    "comment_line": 2,
    "comment_type": "inline",
    "quality_score": 0.8,  # dataset's own — must be ignored
    "is_negative": False,
}


def _row(**overrides: Any) -> dict[str, Any]:
    return {**_VALID_ROW, **overrides}


# ---------------------------------------------------------------------------
# codereview_row_to_pair — filter cases
# ---------------------------------------------------------------------------


def test_row_to_pair_returns_none_for_negative() -> None:
    result = codereview_row_to_pair(_row(is_negative=True))
    assert result is None


def test_row_to_pair_returns_none_for_empty_before_code() -> None:
    assert codereview_row_to_pair(_row(before_code="")) is None
    assert codereview_row_to_pair(_row(before_code="   ")) is None
    assert codereview_row_to_pair(_row(before_code=None)) is None


def test_row_to_pair_returns_none_for_empty_after_code() -> None:
    assert codereview_row_to_pair(_row(after_code="")) is None
    assert codereview_row_to_pair(_row(after_code="   ")) is None
    assert codereview_row_to_pair(_row(after_code=None)) is None


def test_row_to_pair_returns_none_for_empty_reviewer_comment() -> None:
    assert codereview_row_to_pair(_row(reviewer_comment="")) is None
    assert codereview_row_to_pair(_row(reviewer_comment="   ")) is None
    assert codereview_row_to_pair(_row(reviewer_comment=None)) is None


# ---------------------------------------------------------------------------
# codereview_row_to_pair — valid row output
# ---------------------------------------------------------------------------


def test_row_to_pair_produces_correct_sections() -> None:
    pair = codereview_row_to_pair(_VALID_ROW)
    assert pair is not None

    activation = pair["activation_text"]
    assert "## Task" in activation
    assert "## Current Code" in activation
    assert "## Review Feedback" in activation

    teacher = pair["teacher_text"]
    assert "## Revision" in teacher
    # teacher_text extends activation_text
    assert activation in teacher


def test_row_to_pair_quality_score_in_range() -> None:
    pair = codereview_row_to_pair(_VALID_ROW)
    assert pair is not None
    score = pair["quality_score"]
    assert 0.05 <= score <= 1.0


def test_row_to_pair_metadata_source_set() -> None:
    pair = codereview_row_to_pair(_VALID_ROW)
    assert pair is not None
    meta = pair["metadata"]
    assert meta["source"] == "external_codereview"
    assert meta["source_type"] == "external_single_turn"


def test_row_to_pair_ignores_dataset_quality_score() -> None:
    # The dataset provides quality_score=0.8; our scorer computes its own value.
    # They should differ because our scorer applies source_external (0.4) downscaling.
    pair = codereview_row_to_pair(_VALID_ROW)
    assert pair is not None
    assert pair["quality_score"] != _VALID_ROW["quality_score"]


# ---------------------------------------------------------------------------
# ingest_codereview_to_pairs — min_quality_score filtering
# ---------------------------------------------------------------------------


def test_ingest_filters_below_min_quality() -> None:
    # Build rows that will produce known quality spread.
    # Short reviewer_comment (~3 chars) → low quality_score.
    # Rich reviewer_comment (>100 chars) → higher quality_score.
    low_quality_row = _row(reviewer_comment="bad")
    high_quality_row = _row(
        reviewer_comment=(
            "The function should return an explicit value instead of using pass, "
            "which returns None implicitly and can cause AttributeError downstream."
        )
    )

    fake_dataset = [low_quality_row, high_quality_row]

    with patch(
        "model_training.d2l_external.load_codereview_dataset",
        return_value=fake_dataset,
    ):
        # score the two rows to learn their actual computed scores
        low_pair = codereview_row_to_pair(low_quality_row)
        high_pair = codereview_row_to_pair(high_quality_row)
        assert low_pair is not None
        assert high_pair is not None

        low_score = low_pair["quality_score"]
        high_score = high_pair["quality_score"]
        assert low_score < high_score, "test setup: low row must score lower"

        # threshold between the two scores
        threshold = (low_score + high_score) / 2

        pairs = ingest_codereview_to_pairs(min_quality_score=threshold)

    assert len(pairs) == 1
    assert pairs[0]["quality_score"] >= threshold
