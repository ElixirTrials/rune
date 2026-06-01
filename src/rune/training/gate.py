"""Success gate: compare benchmark scores and determine pass/fail."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_BENCHMARKS_PASSING = 4
MIN_IMPROVEMENT = 0.02
MAX_REGRESSION = 0.01

COSINE_MAX = 0.95  # distinct trajectories must diverge below this
DIFF_AGREEMENT_MIN = 0.5


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


@dataclass(frozen=True)
class RetrievalGateResult:
    """Outcome of the content-based retrieval/contrast promotion gate.

    Attributes:
        passed: True if every content criterion was met.
        reasons: Human-readable failure reasons (empty when passed).
    """

    passed: bool
    reasons: tuple[str, ...]


def evaluate_retrieval_gate(probe: dict[str, float]) -> RetrievalGateResult:
    """Content-based promotion gate. Magnitude (scaler_b_absmax) is ignored here."""
    reasons: list[str] = []
    if not probe["real_hit_rate"] > probe["zero_hit_rate"]:
        reasons.append("real_hit_rate <= zero_hit_rate")
    if not probe["real_hit_rate"] > probe["shuffled_hit_rate"]:
        reasons.append("real_hit_rate <= shuffled_hit_rate")
    if not probe["real_hit_rate"] > probe["contradictory_hit_rate"]:
        reasons.append("real_hit_rate <= contradictory_hit_rate")
    if not probe["adapter_cosine"] < COSINE_MAX:
        reasons.append("adapters near-identical (cosine too high)")
    if not probe["diff_agreement"] >= DIFF_AGREEMENT_MIN:
        reasons.append("diff_agreement below threshold")
    return RetrievalGateResult(passed=len(reasons) == 0, reasons=tuple(reasons))
