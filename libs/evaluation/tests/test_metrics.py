"""Tests for evaluation.metrics module.

Green-phase tests for calculate_pass_at_k and run_kill_switch_gate.
Wireframe NotImplementedError tests retained for still-unimplemented stubs.
"""

import math

import pytest
from evaluation.metrics import (
    calculate_pass_at_k,
    compare_adapters,
    evaluate_fitness,
    run_kill_switch_gate,
    run_humaneval_subset,
    score_adapter_quality,
)
from evaluation.metrics import (
    test_generalization as _test_generalization,
)


# ---------------------------------------------------------------------------
# calculate_pass_at_k — green-phase tests
# ---------------------------------------------------------------------------


def test_pass_at_k_perfect_score() -> None:
    """calculate_pass_at_k(1, 1, 1) returns 1.0."""
    assert calculate_pass_at_k(n_samples=1, n_correct=1, k=1) == 1.0


def test_pass_at_k_zero_score() -> None:
    """calculate_pass_at_k(1, 0, 1) returns 0.0."""
    assert calculate_pass_at_k(n_samples=1, n_correct=0, k=1) == 0.0


def test_pass_at_k_direct_rate() -> None:
    """calculate_pass_at_k(100, 85, 1) returns 0.85 (n_samples>>k, pass@1 = c/n)."""
    result = calculate_pass_at_k(n_samples=100, n_correct=85, k=1)
    assert abs(result - 0.85) < 1e-9


def test_pass_at_k_unbiased_estimator() -> None:
    """calculate_pass_at_k(10, 5, 3) returns correct unbiased estimator value.

    Formula: 1 - prod((n-c-i)/(n-i) for i in range(k))
    n=10, c=5, k=3
    term i=0: (10-5-0)/(10-0) = 5/10
    term i=1: (10-5-1)/(10-1) = 4/9
    term i=2: (10-5-2)/(10-2) = 3/8
    prod = (5/10)*(4/9)*(3/8) = 60/720 = 1/12
    pass@3 = 1 - 1/12 = 11/12
    """
    expected = 1 - (5 / 10) * (4 / 9) * (3 / 8)
    result = calculate_pass_at_k(n_samples=10, n_correct=5, k=3)
    assert abs(result - expected) < 1e-9


def test_pass_at_k_raises_on_n_correct_gt_n_samples() -> None:
    """calculate_pass_at_k raises ValueError when n_correct > n_samples."""
    with pytest.raises(ValueError):
        calculate_pass_at_k(n_samples=5, n_correct=10, k=1)


# ---------------------------------------------------------------------------
# run_kill_switch_gate — green-phase tests
# ---------------------------------------------------------------------------


def test_kill_switch_gate_pass() -> None:
    """10% relative improvement >= 5% threshold returns PASS."""
    result = run_kill_switch_gate(baseline_pass1=0.50, adapter_pass1=0.55)
    assert result["verdict"] == "PASS"


def test_kill_switch_gate_fail() -> None:
    """2% relative improvement < 5% threshold returns FAIL."""
    result = run_kill_switch_gate(baseline_pass1=0.50, adapter_pass1=0.51)
    assert result["verdict"] == "FAIL"


def test_kill_switch_gate_zero_baseline() -> None:
    """Zero baseline with non-zero adapter returns PASS without division error."""
    result = run_kill_switch_gate(baseline_pass1=0.0, adapter_pass1=0.05)
    assert result["verdict"] == "PASS"


def test_kill_switch_gate_result_keys() -> None:
    """Result dict contains baseline_pass1, adapter_pass1, relative_delta, verdict."""
    result = run_kill_switch_gate(baseline_pass1=0.50, adapter_pass1=0.55)
    assert "baseline_pass1" in result
    assert "adapter_pass1" in result
    assert "relative_delta" in result
    assert "verdict" in result


# ---------------------------------------------------------------------------
# Wireframe stubs: remaining functions still raise NotImplementedError
# ---------------------------------------------------------------------------


def test_score_adapter_quality_raises_not_implemented() -> None:
    """score_adapter_quality raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="score_adapter_quality"):
        score_adapter_quality("adapter-001", 0.85)


def test_compare_adapters_raises_not_implemented() -> None:
    """compare_adapters raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="compare_adapters"):
        compare_adapters(["adapter-001", "adapter-002"])


def test_test_generalization_raises_not_implemented() -> None:
    """test_generalization raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="test_generalization"):
        _test_generalization("adapter-001")


def test_evaluate_fitness_raises_not_implemented() -> None:
    """evaluate_fitness raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="evaluate_fitness"):
        evaluate_fitness("adapter-001", 0.85)


# ---------------------------------------------------------------------------
# run_humaneval_subset — green-phase tests (placeholder until Task 2)
# These will be replaced in Task 2 with full implementation tests.
# ---------------------------------------------------------------------------


def test_run_humaneval_subset_raises_not_implemented() -> None:
    """run_humaneval_subset raises NotImplementedError with function name."""
    with pytest.raises(NotImplementedError, match="run_humaneval_subset"):
        run_humaneval_subset(adapter_id=None)
