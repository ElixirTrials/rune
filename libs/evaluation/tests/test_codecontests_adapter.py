"""Tests for CodeContestsAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.codecontests import CodeContestsAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "codecontests_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.codecontests.CodeContestsAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list."""
    adapter = CodeContestsAdapter()
    assert len(adapter.load_problems()) > 0


def test_benchmark_id() -> None:
    """benchmark_id is 'codecontests'."""
    assert CodeContestsAdapter.benchmark_id == "codecontests"


def test_problem_fields_populated() -> None:
    """Each problem has non-empty problem_id and prompt."""
    adapter = CodeContestsAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt


def test_score_wrong_returns_verdict() -> None:
    """score() returns a PassVerdict for any generation."""
    adapter = CodeContestsAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "print('wrong')", timeout_s=10)
    assert verdict.problem_id == p.problem_id


def test_score_timeout() -> None:
    """Infinite loop generation returns timed_out=True."""
    adapter = CodeContestsAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "while True: pass", timeout_s=2)
    assert verdict.timed_out is True
