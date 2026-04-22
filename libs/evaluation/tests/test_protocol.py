"""Tests for evaluation.benchmarks.protocol."""

from __future__ import annotations

import pytest
from evaluation.benchmarks.protocol import (
    BenchmarkConfig,
    BenchmarkResult,
    PassVerdict,
    Problem,
)


def test_problem_fields_present() -> None:
    """Problem dataclass stores id, prompt, test_code, and optional metadata."""
    p = Problem(
        problem_id="HumanEval/0",
        prompt="def has_close_elements(numbers, threshold):\n",
        test_code="assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
        entry_point="has_close_elements",
        metadata={"source": "humaneval"},
    )
    assert p.problem_id == "HumanEval/0"
    assert "has_close_elements" in p.prompt
    assert p.entry_point == "has_close_elements"
    assert p.metadata == {"source": "humaneval"}


def test_problem_metadata_defaults_empty() -> None:
    """Problem metadata defaults to empty dict when not provided."""
    p = Problem(
        problem_id="mbpp/1",
        prompt="Write a function.",
        test_code="assert f() == 1",
    )
    assert p.metadata == {}


def test_pass_verdict_passed() -> None:
    """PassVerdict with passed=True stores correct fields."""
    v = PassVerdict(
        problem_id="HumanEval/0",
        passed=True,
        generation="def has_close_elements(): pass",
        error=None,
        timed_out=False,
    )
    assert v.passed is True
    assert v.timed_out is False
    assert v.error is None


def test_pass_verdict_failed_with_error() -> None:
    """PassVerdict with passed=False stores error message."""
    v = PassVerdict(
        problem_id="HumanEval/0",
        passed=False,
        generation="def f(): pass",
        error="AssertionError: ...",
        timed_out=False,
    )
    assert v.passed is False
    assert v.error == "AssertionError: ..."


def test_pass_verdict_timed_out() -> None:
    """PassVerdict with timed_out=True has passed=False."""
    v = PassVerdict(
        problem_id="mbpp/1",
        passed=False,
        generation="",
        error="Execution timed out after 30s",
        timed_out=True,
    )
    assert v.timed_out is True
    assert v.passed is False


def test_benchmark_config_defaults() -> None:
    """BenchmarkConfig has expected defaults."""
    cfg = BenchmarkConfig()
    assert cfg.timeout_s == 30
    assert cfg.max_workers == 4
    assert cfg.max_samples is None
    assert cfg.seed == 42


def test_benchmark_config_custom() -> None:
    """BenchmarkConfig accepts custom values."""
    cfg = BenchmarkConfig(timeout_s=60, max_workers=8, max_samples=100, seed=0)
    assert cfg.timeout_s == 60
    assert cfg.max_workers == 8
    assert cfg.max_samples == 100
    assert cfg.seed == 0


def test_benchmark_result_pass_at_1() -> None:
    """BenchmarkResult.pass_at_1 returns correct fraction."""
    verdicts = [
        PassVerdict(problem_id="a", passed=True, generation="", error=None, timed_out=False),
        PassVerdict(problem_id="b", passed=True, generation="", error=None, timed_out=False),
        PassVerdict(problem_id="c", passed=False, generation="", error="err", timed_out=False),
        PassVerdict(problem_id="d", passed=False, generation="", error="err", timed_out=False),
    ]
    result = BenchmarkResult(benchmark_id="humaneval", verdicts=verdicts)
    assert result.pass_at_1 == 0.5
    assert result.n_problems == 4
    assert result.n_passed == 2


def test_benchmark_result_empty_verdicts_zero() -> None:
    """BenchmarkResult with no verdicts returns pass_at_1 of 0.0."""
    result = BenchmarkResult(benchmark_id="mbpp", verdicts=[])
    assert result.pass_at_1 == 0.0
    assert result.n_problems == 0
