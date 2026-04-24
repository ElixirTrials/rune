"""Tests for DS1000Adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.ds1000 import DS1000Adapter

FIXTURE = Path(__file__).parent / "fixtures" / "ds1000_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.ds1000.DS1000Adapter._fixture_path", FIXTURE
    )


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list."""
    adapter = DS1000Adapter()
    assert len(adapter.load_problems()) > 0


def test_problem_fields_populated() -> None:
    """Each problem has non-empty problem_id and prompt."""
    adapter = DS1000Adapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt


def test_metadata_has_library() -> None:
    """DS-1000 problems include the target library in metadata."""
    adapter = DS1000Adapter()
    for p in adapter.load_problems():
        assert "library" in p.metadata


def test_score_wrong_completion() -> None:
    """score() returns a PassVerdict for any generation."""
    adapter = DS1000Adapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "pass", timeout_s=10)
    assert verdict.problem_id == p.problem_id
