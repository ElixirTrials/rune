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

    # Iterate the union so a baseline benchmark that vanished from new_scores
    # (e.g. the trained model now crashes on it) is counted as a regression
    # instead of being silently skipped.
    for bench in baseline_scores.keys() | new_scores.keys():
        base_score = baseline_scores.get(bench)
        new_score = new_scores.get(bench)
        if base_score is None:
            # New benchmark with no baseline — no delta to judge.
            continue
        if new_score is None:
            regressions[bench] = -base_score
            continue
        delta = new_score - base_score
        if delta >= MIN_IMPROVEMENT:
            improvements[bench] = delta
        elif delta < -MAX_REGRESSION:
            regressions[bench] = delta

    passed = len(improvements) >= MIN_BENCHMARKS_PASSING and len(regressions) == 0
    return GateResult(
        passed=passed,
        passing_benchmarks=len(improvements),
        total_benchmarks=len(baseline_scores.keys() | new_scores.keys()),
        improvements=improvements,
        regressions=regressions,
    )
