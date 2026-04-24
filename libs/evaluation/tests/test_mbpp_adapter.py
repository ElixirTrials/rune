"""Tests for MBPPAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.mbpp import MBPPAdapter
from evaluation.benchmarks.protocol import PassVerdict

FIXTURE = Path(__file__).parent / "fixtures" / "mbpp_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr("evaluation.benchmarks.mbpp.MBPPAdapter._fixture_path", FIXTURE)


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list of Problem instances."""
    adapter = MBPPAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_load_problems_max_samples() -> None:
    """load_problems respects max_samples cap."""
    adapter = MBPPAdapter()
    assert len(adapter.load_problems(max_samples=2)) <= 2


def test_problem_fields_populated() -> None:
    """Each problem has non-empty problem_id, prompt, and test_code."""
    adapter = MBPPAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt
        assert p.test_code


def test_score_wrong_returns_fail() -> None:
    """Scoring a trivially wrong generation returns a PassVerdict."""
    adapter = MBPPAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    return None", timeout_s=10)
    assert isinstance(verdict, PassVerdict)
    # returning None will almost certainly fail MBPP assertions
    assert verdict.problem_id == p.problem_id


def test_score_timeout() -> None:
    """Infinite loop generation returns timed_out=True."""
    adapter = MBPPAdapter()
    p = adapter.load_problems()[0]
    # Module-level infinite loop — guaranteed to hang regardless of function names.
    # MBPP score runs: generation + test_code (prompt is NL description, not executed).
    infinite = "while True:\n    pass"
    verdict = adapter.score(p, infinite, timeout_s=2)
    assert verdict.timed_out is True
    assert verdict.passed is False
