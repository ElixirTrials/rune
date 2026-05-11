"""Tests for McNemar and Wilson CI computation."""

from __future__ import annotations

import pytest

from scripts.paper.statistical_tests import (
    bonferroni_correct,
    mcnemar_test,
    wilson_score_ci,
)


def test_mcnemar_identical_predictions() -> None:
    """Identical predictions yield p=1.0 (no difference)."""
    paired = [(True, True)] * 50 + [(False, False)] * 50
    result = mcnemar_test(paired)
    assert result["p_value"] >= 0.99


def test_mcnemar_all_discordant() -> None:
    """All discordant pairs (one always right, other always wrong) → low p."""
    paired = [(True, False)] * 100
    result = mcnemar_test(paired)
    assert result["p_value"] < 0.001


def test_wilson_ci_bounds() -> None:
    """CI is within [0, 1] and lower <= upper."""
    lower, upper = wilson_score_ci(n_total=100, n_success=70)
    assert 0.0 <= lower <= upper <= 1.0


def test_wilson_ci_perfect_score() -> None:
    """Perfect score has upper bound 1.0."""
    lower, upper = wilson_score_ci(n_total=100, n_success=100)
    assert upper == 1.0
    assert lower > 0.9


def test_bonferroni_correction() -> None:
    """Bonferroni divides alpha by number of comparisons."""
    p_values = [0.01, 0.04, 0.06]
    corrected = bonferroni_correct(p_values, alpha=0.05)
    effective_alpha = 0.05 / 3
    assert corrected["effective_alpha"] == pytest.approx(effective_alpha)
    assert corrected["significant"] == [True, False, False]
