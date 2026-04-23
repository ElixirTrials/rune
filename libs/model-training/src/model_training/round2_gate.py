"""Strict success gate for round-2 vs round-1 benchmark deltas.

Gate (strict):
- round-2 Pass@1 >= round-1 Pass@1 + 2.0% on at least 4 of 6 benchmarks, AND
- no single benchmark regresses by more than 1.0%.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

REQUIRED_BENCHMARKS: tuple[str, ...] = (
    "humaneval",
    "mbpp",
    "apps",
    "bigcodebench",
    "ds_1000",
    "livecodebench",
)
STRICT_IMPROVEMENT_MIN: float = 0.02  # +2.0% absolute Pass@1
STRICT_MAX_REGRESSION: float = 0.01  # -1.0% absolute Pass@1
STRICT_MIN_IMPROVED: int = 4  # out of 6 benchmarks


def evaluate_round2_gate(
    scores: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Apply the strict pass bar to round-1 vs round-2 Pass@1 deltas.

    Args:
        scores: ``{benchmark_id: {"baseline": <r1>, "round2": <r2>}}`` for
            every benchmark in :data:`REQUIRED_BENCHMARKS`.

    Returns:
        Report dict with keys:
        - ``passed`` (bool): gate verdict.
        - ``deltas``: ``{bench: r2 - r1}``.
        - ``improved_count``: count of benchmarks with delta >= STRICT_IMPROVEMENT_MIN.
        - ``max_regression``: largest absolute regression observed
          (``max(0, -delta)``) across all benchmarks.
        - ``reasons``: list[str] describing why the gate failed, or empty
          when ``passed``.

    Raises:
        ValueError: When any required benchmark is missing from ``scores``.
    """
    missing = [b for b in REQUIRED_BENCHMARKS if b not in scores]
    if missing:
        raise ValueError(f"missing required benchmarks: {missing}")

    deltas: dict[str, float] = {}
    for bench in REQUIRED_BENCHMARKS:
        r1 = float(scores[bench]["baseline"])
        r2 = float(scores[bench]["round2"])
        deltas[bench] = r2 - r1

    improved_count = sum(1 for d in deltas.values() if d >= STRICT_IMPROVEMENT_MIN)
    max_regression = max((-d for d in deltas.values() if d < 0), default=0.0)

    reasons: list[str] = []
    if improved_count < STRICT_MIN_IMPROVED:
        reasons.append(
            f"improved {improved_count} benchmark(s) by >= "
            f"{STRICT_IMPROVEMENT_MIN:+.1%}; need {STRICT_MIN_IMPROVED}"
        )
    if max_regression > STRICT_MAX_REGRESSION:
        reasons.append(
            f"max regression {max_regression:.2%} exceeds allowed "
            f"{STRICT_MAX_REGRESSION:.2%}"
        )

    passed = not reasons
    logger.info(
        "Round-2 gate: %s (improved=%d, max_regression=%.2f%%)",
        "PASS" if passed else "FAIL",
        improved_count,
        max_regression * 100,
    )
    return {
        "passed": passed,
        "deltas": deltas,
        "improved_count": improved_count,
        "max_regression": max_regression,
        "reasons": reasons,
    }
