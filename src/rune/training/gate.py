"""Success gate: compare benchmark scores and determine pass/fail."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_BENCHMARKS_PASSING = 4
MIN_IMPROVEMENT = 0.02
MAX_REGRESSION = 0.01


@dataclass(frozen=True)
class GateResult:
    """Outcome of a quality gate evaluation.

    Attributes:
        passed: True if the gate criteria were met.
        passing_benchmarks: Number of benchmarks with sufficient improvement.
        total_benchmarks: Total benchmarks evaluated.
        improvements: Benchmark name → positive delta for improved tasks.
        regressions: Benchmark name → negative delta for regressed tasks.
    """

    passed: bool
    passing_benchmarks: int
    total_benchmarks: int
    improvements: dict[str, float]
    regressions: dict[str, float]


def evaluate_gate(
    baseline_scores: dict[str, float],
    new_scores: dict[str, float],
) -> GateResult:
    """Compare new benchmark scores against a baseline and apply gate thresholds.

    Args:
        baseline_scores: Benchmark name → score before training.
        new_scores: Benchmark name → score after training.

    Returns:
        GateResult indicating pass/fail and per-benchmark deltas.
    """
    improvements: dict[str, float] = {}
    regressions: dict[str, float] = {}

    for bench, new_score in new_scores.items():
        if bench not in baseline_scores:
            continue
        base_score = baseline_scores[bench]
        delta = new_score - base_score
        if delta >= MIN_IMPROVEMENT:
            improvements[bench] = delta
        elif delta < -MAX_REGRESSION:
            regressions[bench] = delta

    passed = len(improvements) >= MIN_BENCHMARKS_PASSING and len(regressions) == 0
    return GateResult(
        passed=passed,
        passing_benchmarks=len(improvements),
        total_benchmarks=len(new_scores),
        improvements=improvements,
        regressions=regressions,
    )
