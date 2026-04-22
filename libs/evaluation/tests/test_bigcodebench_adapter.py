"""Tests for BigCodeBenchAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.bigcodebench import BigCodeBenchAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "bigcodebench_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.bigcodebench.BigCodeBenchAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list."""
    adapter = BigCodeBenchAdapter()
    assert len(adapter.load_problems()) > 0


def test_problem_fields_populated() -> None:
    """Each problem has non-empty problem_id and prompt."""
    adapter = BigCodeBenchAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt


def test_score_wrong_completion_returns_verdict() -> None:
    """A trivially wrong completion returns a PassVerdict (may pass or fail)."""
    adapter = BigCodeBenchAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    return None", timeout_s=10)
    # Score method returns a PassVerdict regardless of result
    assert verdict.problem_id == p.problem_id
    assert isinstance(verdict.passed, bool)


def test_score_syntax_error_returns_fail() -> None:
    """A generation with a syntax error returns passed=False."""
    adapter = BigCodeBenchAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    !!!invalid python!!!", timeout_s=5)
    assert verdict.passed is False
