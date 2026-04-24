"""Tests for HumanEvalAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.humaneval import HumanEvalAdapter
from evaluation.benchmarks.protocol import PassVerdict

FIXTURE = Path(__file__).parent / "fixtures" / "humaneval_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point adapter to local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.humaneval.HumanEvalAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list of Problem instances."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_load_problems_max_samples() -> None:
    """load_problems respects max_samples cap."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems(max_samples=3)
    assert len(problems) <= 3


def test_problem_fields_populated() -> None:
    """Each problem has non-empty problem_id, prompt, test_code, entry_point."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    for p in problems:
        assert p.problem_id.startswith("HumanEval/")
        assert len(p.prompt) > 0
        assert len(p.test_code) > 0
        assert p.entry_point is not None


def test_score_passing_solution() -> None:
    """A trivially wrong solution for a simple problem returns passed=False."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    p = problems[0]
    # `return []` is wrong for most HumanEval problems — we expect failure.
    verdict = adapter.score(p, "    return []", timeout_s=10)
    assert isinstance(verdict, PassVerdict)
    assert verdict.problem_id == p.problem_id
    assert verdict.passed is False


def test_score_correct_identity_solution() -> None:
    """score() returns passed=True when generation passes all assertions."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    p = problems[0]
    canonical = p.metadata.get("canonical_solution")
    if canonical is None:
        pytest.skip("No canonical_solution in fixture metadata")
    verdict = adapter.score(p, canonical, timeout_s=15)
    assert verdict.passed is True


def test_score_timeout_returns_verdict() -> None:
    """score() with an infinite loop returns a PassVerdict with timed_out=True."""
    adapter = HumanEvalAdapter()
    problems = adapter.load_problems()
    p = problems[0]
    infinite_loop = "    while True: pass"
    verdict = adapter.score(p, infinite_loop, timeout_s=2)
    assert isinstance(verdict, PassVerdict)
    assert verdict.timed_out is True
    assert verdict.passed is False
