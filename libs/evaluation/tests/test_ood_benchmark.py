"""Tests for evaluation.ood_benchmark — OOD benchmark and generalization delta."""

import json
from pathlib import Path

from evaluation.ood_benchmark import compute_generalization_delta, run_ood_benchmark

OOD_JSON = (
    Path(__file__).parent.parent / "src" / "evaluation" / "data" / "ood_tasks.json"
)


def test_ood_json_exists() -> None:
    assert OOD_JSON.exists()


def test_ood_json_has_10_tasks() -> None:
    with OOD_JSON.open() as f:
        tasks = json.load(f)
    assert len(tasks) == 10


def test_generalization_delta_positive() -> None:
    delta = compute_generalization_delta(in_dist_rate=0.7, ood_rate=0.9)
    assert abs(delta - 0.2) < 1e-9


def test_generalization_delta_negative() -> None:
    delta = compute_generalization_delta(in_dist_rate=0.9, ood_rate=0.7)
    assert abs(delta - (-0.2)) < 1e-9


def test_run_ood_benchmark_with_completions() -> None:
    with OOD_JSON.open() as f:
        tasks = json.load(f)
    task = tasks[0]
    completions = {task["task_id"]: task["canonical_solution"]}
    result = run_ood_benchmark(adapter_id=None, completions=completions)
    assert "ood_pass_rate" in result
    assert "task_results" in result
    assert len(result["task_results"]) == 1
    assert result["task_results"][0]["passed"] is True


def test_run_ood_benchmark_empty_completions() -> None:
    result = run_ood_benchmark(adapter_id=None, completions={})
    assert result["ood_pass_rate"] == 0.0
    assert result["task_results"] == []
