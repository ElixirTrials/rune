"""Golden-file tests for the Pass@1 aggregation function."""

from __future__ import annotations

import pytest
from evaluation.benchmarks.aggregator import pass_at_1_from_verdicts
from evaluation.benchmarks.protocol import PassVerdict


def _verdict(problem_id: str, passed: bool) -> PassVerdict:
    return PassVerdict(
        problem_id=problem_id,
        passed=passed,
        generation="",
        error=None if passed else "err",
        timed_out=False,
    )


def test_all_passed() -> None:
    """All 5 verdicts passed → pass@1 = 1.0."""
    verdicts = [_verdict(f"p{i}", True) for i in range(5)]
    assert pass_at_1_from_verdicts(verdicts) == 1.0


def test_none_passed() -> None:
    """All 5 verdicts failed → pass@1 = 0.0."""
    verdicts = [_verdict(f"p{i}", False) for i in range(5)]
    assert pass_at_1_from_verdicts(verdicts) == 0.0


def test_half_passed() -> None:
    """3 of 6 passed → pass@1 = 0.5."""
    verdicts = [_verdict(f"p{i}", i < 3) for i in range(6)]
    assert pass_at_1_from_verdicts(verdicts) == 0.5


def test_empty_returns_zero() -> None:
    """No verdicts → pass@1 = 0.0 (no ZeroDivisionError)."""
    assert pass_at_1_from_verdicts([]) == 0.0


def test_single_pass() -> None:
    """Single passing verdict → pass@1 = 1.0."""
    assert pass_at_1_from_verdicts([_verdict("x", True)]) == 1.0


def test_single_fail() -> None:
    """Single failing verdict → pass@1 = 0.0."""
    assert pass_at_1_from_verdicts([_verdict("x", False)]) == 0.0


def test_golden_4_of_5() -> None:
    """Golden: 4 of 5 pass → 0.8 exactly."""
    verdicts = [_verdict(f"p{i}", i < 4) for i in range(5)]
    result = pass_at_1_from_verdicts(verdicts)
    assert abs(result - 0.8) < 1e-9


def test_timed_out_counts_as_fail() -> None:
    """Timed-out verdicts count as failures in the aggregation."""
    verdicts = [
        PassVerdict(problem_id="a", passed=False, generation="", error="timeout", timed_out=True),
        PassVerdict(problem_id="b", passed=True, generation="ok", error=None, timed_out=False),
    ]
    assert pass_at_1_from_verdicts(verdicts) == 0.5
