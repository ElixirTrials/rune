"""Tests for the per-datapoint quality heuristic."""

from __future__ import annotations

import pytest

from model_training.d2l_quality import (
    QualityWeightConfig,
    classify_causal_link,
    is_url_only,
    score_episode_quality,
    score_external_quality,
)


# ---------------------------------------------------------------------------
# is_url_only
# ---------------------------------------------------------------------------


class TestIsUrlOnly:
    def test_single_url(self) -> None:
        assert is_url_only("https://circleci.com/gh/org/repo/123") is True

    def test_multiple_urls(self) -> None:
        assert is_url_only("https://a.com https://b.com") is True

    def test_url_with_text(self) -> None:
        assert is_url_only("See https://a.com for details") is False

    def test_plain_text(self) -> None:
        assert is_url_only("rename foo to bar") is False

    def test_empty(self) -> None:
        assert is_url_only("") is False

    def test_whitespace_only(self) -> None:
        assert is_url_only("   ") is False


# ---------------------------------------------------------------------------
# classify_causal_link
# ---------------------------------------------------------------------------


class TestClassifyCausalLink:
    def test_entity_overlap(self) -> None:
        fb = "rename parse_response to handle_response"
        diff = "+def handle_response(data):\n+    return data"
        assert classify_causal_link(fb, diff) == "entity_overlap"

    def test_no_overlap(self) -> None:
        fb = "this approach is fragile and should be reworked"
        diff = "+def validate(x):\n+    if x < 0: raise ValueError"
        assert classify_causal_link(fb, diff) == "no_overlap"

    def test_url_only_feedback(self) -> None:
        fb = "https://circleci.com/gh/org/repo/123"
        diff = "+def foo(): pass"
        assert classify_causal_link(fb, diff) == "url_only"

    def test_short_identifiers_excluded(self) -> None:
        fb = "fix it"
        diff = "+x = it + 1"
        # "it" is only 2 chars, below the 3-char identifier threshold
        assert classify_causal_link(fb, diff) == "no_overlap"


# ---------------------------------------------------------------------------
# score_episode_quality
# ---------------------------------------------------------------------------


class TestScoreEpisodeQuality:
    def test_best_case_trajectory(self) -> None:
        score = score_episode_quality(
            feedback_body="The parse_response function should handle None values gracefully and return an empty dict instead of raising",
            action_diff="+def parse_response(data):\n+    if data is None:\n+        return {}",
            is_ep0=False,
        )
        assert score == pytest.approx(1.0)

    def test_ep0_skips_causal_factor(self) -> None:
        score = score_episode_quality(
            feedback_body="Implement a REST API for user management with CRUD operations",
            action_diff="+class UserController:\n+    pass",
            is_ep0=True,
        )
        cfg = QualityWeightConfig()
        # ep0: source=1.0, causal=1.0(skip), feedback=0.7(62 chars=moderate), prop=1.0
        assert score == pytest.approx(cfg.feedback_moderate_factor)
        # Verify ep0 is NOT penalized by causal factor (no-overlap would give 0.28)
        non_ep0 = score_episode_quality(
            feedback_body="Implement a REST API for user management with CRUD operations",
            action_diff="+class UserController:\n+    pass",
            is_ep0=False,
        )
        assert score > non_ep0

    def test_no_overlap_moderate_feedback(self) -> None:
        score = score_episode_quality(
            feedback_body="this looks wrong, please fix",
            action_diff="+def validate(x): return x > 0",
            is_ep0=False,
        )
        cfg = QualityWeightConfig()
        # source=1.0, causal=0.4(no_overlap), feedback=0.7(28 chars), prop=1.0
        assert score == pytest.approx(
            cfg.source_trajectory * cfg.causal_no_overlap * cfg.feedback_moderate_factor
        )

    def test_url_only_hits_floor(self) -> None:
        score = score_episode_quality(
            feedback_body="https://circleci.com/build/123",
            action_diff="+x = 1",
            is_ep0=False,
        )
        cfg = QualityWeightConfig()
        assert score == cfg.floor

    def test_proportionality_penalty(self) -> None:
        score = score_episode_quality(
            feedback_body="fix",
            action_diff="+" * 6000,
            is_ep0=False,
        )
        score_no_penalty = score_episode_quality(
            feedback_body="fix",
            action_diff="+" * 100,
            is_ep0=False,
        )
        assert score < score_no_penalty

    def test_floor_respected(self) -> None:
        cfg = QualityWeightConfig(floor=0.1)
        score = score_episode_quality(
            feedback_body="https://ci.example.com",
            action_diff="+" * 6000,
            is_ep0=False,
            config=cfg,
        )
        assert score >= cfg.floor

    def test_custom_config(self) -> None:
        cfg = QualityWeightConfig(causal_no_overlap=0.8)
        score = score_episode_quality(
            feedback_body="this looks wrong, please fix",
            action_diff="+def validate(x): return x > 0",
            is_ep0=False,
            config=cfg,
        )
        default_score = score_episode_quality(
            feedback_body="this looks wrong, please fix",
            action_diff="+def validate(x): return x > 0",
            is_ep0=False,
        )
        assert score > default_score


# ---------------------------------------------------------------------------
# score_external_quality
# ---------------------------------------------------------------------------


class TestScoreExternalQuality:
    def test_good_external_review(self) -> None:
        # 99 chars — just under the 100-char rich threshold
        fb = "The validate_input function should check for None before accessing .strip() to avoid AttributeError"
        score = score_external_quality(
            feedback_body=fb,
            before_code="def validate_input(s):\n    return s.strip()",
            after_code="def validate_input(s):\n    if s is None:\n        return ''\n    return s.strip()",
        )
        cfg = QualityWeightConfig()
        # source=0.4, feedback=0.7(99 chars=moderate), prop=1.0
        assert score == pytest.approx(cfg.source_external * cfg.feedback_moderate_factor)

    def test_external_always_below_trajectory(self) -> None:
        fb = "rename parse_response to handle_response — it does more than parse"
        diff = "+def handle_response(data):\n+    return data"
        ext = score_external_quality(
            feedback_body=fb, before_code="old", after_code=diff,
        )
        traj = score_episode_quality(
            feedback_body=fb, action_diff=diff, is_ep0=False,
        )
        assert ext < traj

    def test_short_feedback_large_diff_penalized(self) -> None:
        score = score_external_quality(
            feedback_body="fix",
            before_code="x",
            after_code="+" * 6000,
        )
        cfg = QualityWeightConfig()
        assert score == pytest.approx(
            max(
                cfg.floor,
                cfg.source_external * cfg.feedback_short_factor * cfg.proportionality_penalty,
            )
        )

    def test_floor_respected(self) -> None:
        score = score_external_quality(
            feedback_body="https://example.com",
            before_code="x",
            after_code="+" * 6000,
        )
        assert score >= QualityWeightConfig().floor
