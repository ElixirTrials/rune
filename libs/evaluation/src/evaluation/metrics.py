"""Evaluation metrics for adapter benchmarking and fitness scoring.

Provides functions for running benchmarks (HumanEval), calculating metrics
(Pass@k, quality scores), comparing adapters, testing generalization,
and computing evolutionary fitness for the evolution operator.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Optional

from evaluation.utils import safe_subprocess_run

_DATA_DIR = Path(__file__).parent / "data"


def calculate_pass_at_k(n_samples: int, n_correct: int, k: int = 1) -> float:
    """Calculate the Pass@k metric for code generation evaluation.

    Computes the probability that at least one of k sampled solutions is
    correct, using the unbiased estimator from the HumanEval paper
    (Chen et al., 2021). This avoids the high-variance naive estimator.

    Formula: 1 - prod((n-c-i)/(n-i) for i in range(k))

    Args:
        n_samples: Total number of samples generated per problem.
        n_correct: Number of correct samples out of n_samples.
        k: Number of attempts allowed for the Pass@k metric. Defaults to 1.

    Returns:
        Pass@k probability as a float between 0.0 and 1.0.

    Raises:
        ValueError: If n_correct > n_samples.

    Example:
        >>> score = calculate_pass_at_k(n_samples=100, n_correct=85, k=1)
        >>> score
        0.85
    """
    if n_correct > n_samples:
        raise ValueError(
            f"n_correct ({n_correct}) cannot be greater than n_samples ({n_samples})"
        )
    if n_correct == n_samples:
        return 1.0
    if n_samples - n_correct < k:
        return 1.0
    # Unbiased estimator: 1 - prod((n-c-i)/(n-i) for i in range(k))
    product = math.prod((n_samples - n_correct - i) / (n_samples - i) for i in range(k))
    return 1.0 - product


def run_kill_switch_gate(
    baseline_pass1: float,
    adapter_pass1: float,
    threshold: float = 0.05,
) -> dict[str, object]:
    """Compare baseline vs adapter Pass@1 scores and return a PASS/FAIL verdict.

    Computes the relative improvement of the adapter over the baseline and
    returns PASS if the improvement meets the threshold, FAIL otherwise.

    Args:
        baseline_pass1: Pass@1 score for the baseline model (no adapter).
        adapter_pass1: Pass@1 score for the adapter model.
        threshold: Minimum required relative improvement (default 0.05 = 5%).

    Returns:
        Dictionary with:
            - "baseline_pass1": float
            - "adapter_pass1": float
            - "relative_delta": float, relative improvement over baseline
            - "verdict": str, "PASS" or "FAIL"

    Example:
        >>> result = run_kill_switch_gate(0.50, 0.55)
        >>> result["verdict"]
        'PASS'
    """
    # 1e-9 prevents division-by-zero when baseline is 0.0
    relative_delta = (adapter_pass1 - baseline_pass1) / max(baseline_pass1, 1e-9)
    verdict = "PASS" if adapter_pass1 >= baseline_pass1 * (1 + threshold) else "FAIL"

    print(
        f"Kill-switch gate result: {verdict}\n"
        f"  Baseline  Pass@1: {baseline_pass1:.4f}\n"
        f"  Adapter   Pass@1: {adapter_pass1:.4f}\n"
        f"  Relative delta:   {relative_delta:+.2%} (threshold: {threshold:+.0%})"
    )

    return {
        "baseline_pass1": baseline_pass1,
        "adapter_pass1": adapter_pass1,
        "relative_delta": relative_delta,
        "verdict": verdict,
    }


def run_humaneval_subset(
    adapter_id: Optional[str],
    subset_size: int = 20,
    model: str = "Qwen/Qwen3.5-9B",
    completions: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Run a HumanEval benchmark subset to evaluate an adapter.

    Loads tasks from the bundled 20-task HumanEval subset JSON and evaluates
    them using the provided completions. For each task, concatenates
    prompt + completion + test + check(entry_point), writes to a temp script,
    and executes via subprocess. Exit code 0 = passed.

    Args:
        adapter_id: UUID of the adapter to test (None = baseline, no adapter).
            Currently informational; inference wiring happens at a higher level.
        subset_size: Ignored — always uses the fixed 20-task bundled subset.
        model: Base model name (informational only, not used for inference here).
        completions: Dict mapping task_id -> completion string. Only tasks with
            an entry in this dict are evaluated. If None or empty, returns empty
            results.

    Returns:
        Dictionary with benchmark results including:
            - "pass_count": int, number of tasks passed
            - "fail_count": int, number of tasks failed
            - "pass_rate": float, fraction of tasks passed (0.0 to 1.0)
            - "task_results": list of per-task result dicts with task_id, passed
            - "summary": str, human-readable summary

    Raises:
        NotImplementedError: If completions are not provided (inference not wired).

    Example:
        >>> completions = {"HumanEval/0": "    return []"}
        >>> results = run_humaneval_subset(adapter_id=None, completions=completions)
        >>> results["pass_count"]
        0
    """
    if completions is None:
        raise NotImplementedError("run_humaneval_subset is not yet implemented.")

    # Load bundled task data
    subset_path = _DATA_DIR / "humaneval_subset.json"
    with subset_path.open() as f:
        all_tasks: list[dict[str, str]] = json.load(f)

    # Build lookup by task_id
    task_map = {t["task_id"]: t for t in all_tasks}

    task_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for task_id, completion in completions.items():
            task = task_map.get(task_id)
            if task is None:
                continue

            # Build executable script: prompt + completion + test + check(entry_point)
            script = (
                task["prompt"]
                + completion
                + "\n"
                + task["test"]
                + f"\ncheck({task['entry_point']})\n"
            )

            script_path = Path(tmpdir) / f"{task_id.replace('/', '_')}.py"
            script_path.write_text(script)

            passed = safe_subprocess_run(script_path, cwd=tmpdir)
            task_results.append({"task_id": task_id, "passed": passed})

    pass_count = sum(1 for r in task_results if r["passed"])
    fail_count = len(task_results) - pass_count
    total = len(task_results)
    pass_rate = pass_count / total if total > 0 else 0.0

    summary = (
        f"HumanEval subset: {pass_count}/{total} passed "
        f"(pass_rate={pass_rate:.2%}, adapter_id={adapter_id})"
    )
    print(summary)

    return {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "task_results": task_results,
        "summary": summary,
    }


def score_adapter_quality(
    adapter_id: str,
    pass_rate: float,
    generalization_delta: float | None = None,
) -> float:
    """Compute an overall quality score for an adapter.

    Aggregates the adapter's benchmark pass rate with its generalization
    performance (if available) into a single scalar quality score. When
    generalization data is absent, quality is derived from pass rate alone.

    Args:
        adapter_id: UUID of the adapter to score.
        pass_rate: Fraction of benchmark tasks passed, in range 0.0 to 1.0.
        generalization_delta: Optional difference between in-distribution and
            out-of-distribution (OOD) performance. Positive values indicate
            the adapter generalizes well; negative values indicate overfitting.
            Defaults to None (not measured).

    Returns:
        Quality score as a float between 0.0 and 1.0. Higher values indicate
        a better overall adapter quality.

    Raises:
        NotImplementedError: score_adapter_quality is not yet implemented.

    Example:
        >>> score = score_adapter_quality("adapter-001", pass_rate=0.85)
        >>> score
        0.85
        >>> score_with_gen = score_adapter_quality(
        ...     "adapter-001", 0.85, generalization_delta=0.1,
        ... )
        >>> score_with_gen > score
        True
    """
    if generalization_delta is not None:
        quality = min(pass_rate + 0.1 * max(generalization_delta, 0), 1.0)
    else:
        quality = pass_rate
    return quality


def compare_adapters(
    adapter_ids: list[str],
    benchmark: str = "humaneval",
) -> dict[str, Any]:
    """Compare multiple adapters head-to-head on a benchmark.

    Evaluates each adapter in the list on the specified benchmark and produces
    a comparative report with per-adapter scores, rankings, and a summary of
    which adapter performs best.

    Args:
        adapter_ids: List of adapter UUIDs to compare. Must contain at least
            two adapter IDs.
        benchmark: Benchmark name to use for comparison. Currently supports
            "humaneval". Defaults to "humaneval".

    Returns:
        Dictionary with comparison results including:
            - "scores": dict mapping adapter_id to its benchmark score
            - "rankings": list of adapter_ids sorted best-to-worst
            - "best_adapter": str, UUID of the top-performing adapter
            - "summary": str, human-readable comparison summary

    Raises:
        NotImplementedError: compare_adapters is not yet implemented.

    Example:
        >>> results = compare_adapters(["adapter-001", "adapter-002"])
        >>> results["best_adapter"] in ["adapter-001", "adapter-002"]
        True
        >>> results["best_adapter"] in results["rankings"]
        True
    """
    if len(adapter_ids) < 2:
        raise ValueError("compare_adapters requires at least 2 adapter IDs")

    # Without live inference, return a stub comparison based on adapter order
    scores: dict[str, float] = {}
    for i, aid in enumerate(adapter_ids):
        scores[aid] = 1.0 / (i + 1)  # Placeholder scoring

    rankings = sorted(scores, key=lambda x: scores[x], reverse=True)
    best = rankings[0]
    summary = f"Compared {len(adapter_ids)} adapters on {benchmark}; best={best}"

    return {
        "scores": scores,
        "rankings": rankings,
        "best_adapter": best,
        "summary": summary,
    }


def test_generalization(
    adapter_id: str,
    in_distribution_tasks: list[str] | None = None,
    ood_tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Test whether an adapter generalizes beyond its training distribution.

    Evaluates the adapter on both in-distribution tasks (matching the training
    data distribution) and out-of-distribution (OOD) tasks to measure how well
    the adapter generalizes to novel problems.

    Args:
        adapter_id: UUID of the adapter to evaluate.
        in_distribution_tasks: Optional list of task IDs matching the adapter's
            training distribution. If None, uses a default in-distribution set.
        ood_tasks: Optional list of out-of-distribution task IDs to test
            generalization on. If None, uses a default OOD task set.

    Returns:
        Dictionary with generalization results including:
            - "in_distribution_score": float, performance on training-distribution tasks
            - "ood_score": float, performance on out-of-distribution tasks
            - "generalization_delta": float, difference (in_distribution - ood)
            - "generalizes": bool, True if generalization_delta is within threshold

    Raises:
        NotImplementedError: test_generalization is not yet implemented.

    Example:
        >>> results = test_generalization("adapter-001")
        >>> results["generalization_delta"]
        0.05
        >>> results["generalizes"]
        True
    """
    in_dist_score = 0.8  # Placeholder until live inference wired
    ood_score = 0.6
    gen_delta = in_dist_score - ood_score
    generalizes = abs(gen_delta) <= 0.2

    return {
        "adapter_id": adapter_id,
        "in_distribution_score": in_dist_score,
        "ood_score": ood_score,
        "generalization_delta": gen_delta,
        "generalizes": generalizes,
    }


def evaluate_fitness(
    adapter_id: str,
    pass_rate: float,
    diversity_score: float = 0.0,
) -> float:
    """Calculate evolutionary fitness score for the evolution operator.

    Computes a composite fitness score used by the evolutionary algorithm to
    rank and select adapters for mutation and crossover. Balances raw
    performance (pass_rate) with adapter diversity to avoid population collapse.

    Args:
        adapter_id: UUID of the adapter to evaluate.
        pass_rate: Fraction of benchmark tasks passed, in range 0.0 to 1.0.
        diversity_score: Adapter uniqueness metric representing how different
            this adapter is from others in the current population. Range 0.0
            to 1.0; higher values indicate more unique adapters. Defaults to 0.0.

    Returns:
        Fitness score as a float between 0.0 and 1.0. Higher values indicate
        adapters more likely to be selected for the next evolutionary generation.

    Raises:
        NotImplementedError: evaluate_fitness is not yet implemented.

    Example:
        >>> fitness = evaluate_fitness(
        ...     "adapter-001", pass_rate=0.85, diversity_score=0.3,
        ... )
        >>> fitness
        0.795
        >>> low_diversity = evaluate_fitness(
        ...     "adapter-001", pass_rate=0.85, diversity_score=0.0,
        ... )
        >>> low_diversity < fitness
        True
    """
    fitness = 0.7 * pass_rate + 0.3 * diversity_score
    return fitness
