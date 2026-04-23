"""CPU-only unit tests for evaluate_round2_gate (no benchmark runs)."""

from __future__ import annotations

import pytest
from model_training.round2_gate import (
    STRICT_IMPROVEMENT_MIN,
    STRICT_MAX_REGRESSION,
    STRICT_MIN_IMPROVED,
    evaluate_round2_gate,
)


def _make_scores(**pairs: tuple[float, float]) -> dict[str, dict[str, float]]:
    """{bench: {"baseline": x, "round2": y}}."""
    return {k: {"baseline": v[0], "round2": v[1]} for k, v in pairs.items()}


def test_gate_passes_when_strict_criteria_met() -> None:
    """4 of 6 benchmarks improved by >= 2%, no regression > 1%."""
    scores = _make_scores(
        humaneval=(0.60, 0.64),  # +4.0 ✓ improved
        mbpp=(0.50, 0.53),  # +3.0 ✓ improved
        apps=(0.20, 0.225),  # +2.5 ✓ improved
        bigcodebench=(0.35, 0.372),  # +2.2 ✓ improved
        ds_1000=(0.40, 0.405),  # +0.5 (not ≥ 2, not regress)
        livecodebench=(0.15, 0.145),  # -0.5 (regression < 1, acceptable)
    )
    report = evaluate_round2_gate(scores)
    assert report["passed"] is True
    assert report["improved_count"] == 4
    assert report["max_regression"] == pytest.approx(0.005)


def test_gate_fails_when_too_few_improved() -> None:
    """Only 3 improved by ≥ 2% → gate fails."""
    scores = _make_scores(
        humaneval=(0.60, 0.64),  # +4.0 ✓
        mbpp=(0.50, 0.53),  # +3.0 ✓
        apps=(0.20, 0.225),  # +2.5 ✓
        bigcodebench=(0.35, 0.36),  # +1.0 (not enough)
        ds_1000=(0.40, 0.40),
        livecodebench=(0.15, 0.15),
    )
    report = evaluate_round2_gate(scores)
    assert report["passed"] is False
    assert report["improved_count"] == 3


def test_gate_fails_on_excess_regression() -> None:
    """Any regression > 1% fails the gate even if 4+ benchmarks improved."""
    scores = _make_scores(
        humaneval=(0.60, 0.64),  # ✓
        mbpp=(0.50, 0.53),  # ✓
        apps=(0.20, 0.225),  # ✓
        bigcodebench=(0.35, 0.372),  # ✓
        ds_1000=(0.40, 0.40),
        livecodebench=(0.15, 0.135),  # -1.5 ✗ excess regression
    )
    report = evaluate_round2_gate(scores)
    assert report["passed"] is False
    assert report["max_regression"] == pytest.approx(0.015)


def test_gate_thresholds_are_reasonable_constants() -> None:
    """Sanity-check the strict thresholds."""
    assert STRICT_IMPROVEMENT_MIN == pytest.approx(0.02)
    assert STRICT_MAX_REGRESSION == pytest.approx(0.01)
    assert STRICT_MIN_IMPROVED == 4


def test_gate_rejects_unknown_benchmarks() -> None:
    """Gate requires the canonical 6 benchmark keys."""
    scores = _make_scores(humaneval=(0.5, 0.55))
    with pytest.raises(ValueError, match="missing required benchmarks"):
        evaluate_round2_gate(scores)
