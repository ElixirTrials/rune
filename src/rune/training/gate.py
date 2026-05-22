from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_BENCHMARKS_PASSING = 4
MIN_IMPROVEMENT = 2.0
MAX_REGRESSION = 1.0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    passing_benchmarks: int
    total_benchmarks: int
    improvements: dict[str, float]
    regressions: dict[str, float]


def evaluate_gate(
    baseline_scores: dict[str, float],
    new_scores: dict[str, float],
) -> GateResult:
    improvements: dict[str, float] = {}
    regressions: dict[str, float] = {}

    for bench, new_score in new_scores.items():
        base_score = baseline_scores.get(bench, 0.0)
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
