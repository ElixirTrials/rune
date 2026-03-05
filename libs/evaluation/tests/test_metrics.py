"""Tests for evaluation.metrics module.

Green-phase tests for calculate_pass_at_k, run_kill_switch_gate, and
run_humaneval_subset. Wireframe NotImplementedError tests retained for
still-unimplemented stubs.
"""

import json
from pathlib import Path

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
# run_humaneval_subset — green-phase tests
# ---------------------------------------------------------------------------

HUMANEVAL_JSON = (
    Path(__file__).parent.parent
    / "src"
    / "evaluation"
    / "data"
    / "humaneval_subset.json"
)


def test_humaneval_json_exists() -> None:
    """humaneval_subset.json file exists in the data directory."""
    assert HUMANEVAL_JSON.exists(), f"Expected JSON at {HUMANEVAL_JSON}"


def test_humaneval_json_has_20_tasks() -> None:
    """humaneval_subset.json contains exactly 20 tasks."""
    with HUMANEVAL_JSON.open() as f:
        tasks = json.load(f)
    assert len(tasks) == 20, f"Expected 20 tasks, got {len(tasks)}"


def test_humaneval_json_task_fields() -> None:
    """Each task has required fields: task_id, prompt, canonical_solution, test, entry_point."""
    with HUMANEVAL_JSON.open() as f:
        tasks = json.load(f)
    required = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
    for task in tasks:
        missing = required - task.keys()
        assert not missing, f"Task {task.get('task_id', '?')} missing fields: {missing}"


def test_humaneval_json_contains_task_0() -> None:
    """humaneval_subset.json contains HumanEval/0."""
    with HUMANEVAL_JSON.open() as f:
        tasks = json.load(f)
    task_ids = {t["task_id"] for t in tasks}
    assert "HumanEval/0" in task_ids


def test_run_humaneval_subset_baseline_returns_expected_keys() -> None:
    """run_humaneval_subset returns dict with pass_count, fail_count, pass_rate, task_results."""
    # Use canonical solutions as completions for a subset
    with HUMANEVAL_JSON.open() as f:
        tasks = json.load(f)
    # Just pass one task via completions
    first_task = tasks[0]
    completions = {first_task["task_id"]: first_task["canonical_solution"]}
    result = run_humaneval_subset(adapter_id=None, completions=completions)
    assert "pass_count" in result
    assert "fail_count" in result
    assert "pass_rate" in result
    assert "task_results" in result


def test_run_humaneval_subset_task_results_structure() -> None:
    """task_results has one entry per task with task_id and passed boolean."""
    with HUMANEVAL_JSON.open() as f:
        tasks = json.load(f)
    first_task = tasks[0]
    completions = {first_task["task_id"]: first_task["canonical_solution"]}
    result = run_humaneval_subset(adapter_id=None, completions=completions)
    assert len(result["task_results"]) == 1
    tr = result["task_results"][0]
    assert "task_id" in tr
    assert "passed" in tr
    assert isinstance(tr["passed"], bool)


def test_run_humaneval_subset_canonical_solution_passes() -> None:
    """A task with correct canonical_solution passes the execution pipeline."""
    with HUMANEVAL_JSON.open() as f:
        tasks = json.load(f)
    # HumanEval/0: has_close_elements — canonical solution should pass
    task = next(t for t in tasks if t["task_id"] == "HumanEval/0")
    completions = {task["task_id"]: task["canonical_solution"]}
    result = run_humaneval_subset(adapter_id=None, completions=completions)
    assert result["pass_count"] >= 1
    passed_task = next(
        t for t in result["task_results"] if t["task_id"] == "HumanEval/0"
    )
    assert passed_task["passed"] is True


def test_run_humaneval_subset_wrong_completion_fails() -> None:
    """An intentionally wrong completion results in a failed task."""
    with HUMANEVAL_JSON.open() as f:
        tasks = json.load(f)
    task = tasks[0]
    # Deliberately wrong: always returns empty list
    completions = {task["task_id"]: "    return []"}
    result = run_humaneval_subset(adapter_id=None, completions=completions)
    failed_task = next(
        t for t in result["task_results"] if t["task_id"] == task["task_id"]
    )
    assert failed_task["passed"] is False
