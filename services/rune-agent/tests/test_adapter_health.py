"""Tests for adapter health monitoring."""

from __future__ import annotations

import pytest

from rune_agent.adapter_health import (
    check_health,
    compute_cosine_similarity,
    compute_norm_ratio,
    compute_output_repetition,
)


def test_cosine_similarity_identical():
    weights = {"layer.0.A": [1.0, 2.0, 3.0], "layer.0.B": [4.0, 5.0, 6.0]}
    sim = compute_cosine_similarity(weights, weights)
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal():
    w1 = {"layer.0.A": [1.0, 0.0]}
    w2 = {"layer.0.A": [0.0, 1.0]}
    sim = compute_cosine_similarity(w1, w2)
    assert sim == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity_none_previous():
    w1 = {"layer.0.A": [1.0, 2.0]}
    sim = compute_cosine_similarity(w1, None)
    assert sim == 0.0


def test_norm_ratio_stable():
    weights = {"layer.0.A": [3.0, 4.0]}
    first_norm = 5.0
    ratio = compute_norm_ratio(weights, first_norm)
    assert ratio == pytest.approx(1.0, abs=1e-5)


def test_norm_ratio_collapsed():
    weights = {"layer.0.A": [0.01, 0.01]}
    first_norm = 5.0
    ratio = compute_norm_ratio(weights, first_norm)
    assert ratio < 0.1


def test_norm_ratio_no_first():
    weights = {"layer.0.A": [3.0, 4.0]}
    ratio = compute_norm_ratio(weights, None)
    assert ratio == 1.0


def test_output_repetition_identical():
    text = "the quick brown fox jumps over the lazy dog again and again"
    rep = compute_output_repetition(text, text)
    assert rep == pytest.approx(1.0)


def test_output_repetition_no_overlap():
    t1 = "alpha beta gamma delta epsilon zeta eta theta"
    t2 = "one two three four five six seven eight"
    rep = compute_output_repetition(t1, t2)
    assert rep == 0.0


def test_output_repetition_empty():
    assert compute_output_repetition("", "hello world") == 0.0
    assert compute_output_repetition("hello world", "") == 0.0


def test_check_health_healthy():
    h = check_health(
        cosine_sim=0.5, norm_ratio=1.0, output_repetition=0.2,
        consecutive_high_similarity=0,
    )
    assert h.is_collapsed is False
    assert h.collapse_reason is None


def test_check_health_norm_collapse():
    h = check_health(
        cosine_sim=0.5, norm_ratio=0.05, output_repetition=0.2,
        consecutive_high_similarity=0,
    )
    assert h.is_collapsed is True
    assert "norm_collapse" in h.collapse_reason


def test_check_health_norm_explosion():
    h = check_health(
        cosine_sim=0.5, norm_ratio=15.0, output_repetition=0.2,
        consecutive_high_similarity=0,
    )
    assert h.is_collapsed is True
    assert "norm_explosion" in h.collapse_reason


def test_check_health_cosine_collapse_needs_consecutive():
    h1 = check_health(
        cosine_sim=0.97, norm_ratio=1.0, output_repetition=0.2,
        consecutive_high_similarity=0,
    )
    assert h1.is_collapsed is False

    h2 = check_health(
        cosine_sim=0.97, norm_ratio=1.0, output_repetition=0.2,
        consecutive_high_similarity=1,
    )
    assert h2.is_collapsed is True
    assert "cosine_collapse" in h2.collapse_reason


def test_check_health_output_repetition():
    h = check_health(
        cosine_sim=0.5, norm_ratio=1.0, output_repetition=0.85,
        consecutive_high_similarity=0,
    )
    assert h.is_collapsed is True
    assert "output_repetition" in h.collapse_reason
