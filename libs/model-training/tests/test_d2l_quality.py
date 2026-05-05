"""Tests for the per-datapoint quality heuristic."""

from __future__ import annotations

import pytest
from model_training.d2l_quality import (
    QualityWeightConfig,
    classify_causal_link,
    compute_causal_density,
    compute_diff_focus,
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
        fb = (
            "The parse_response function should handle None values"
            " gracefully and return an empty dict instead of raising"
        )
        diff = "+def parse_response(data):\n+    if data is None:\n+        return {}"
        score = score_episode_quality(
            feedback_body=fb,
            action_diff=diff,
            is_ep0=False,
        )
        assert score == pytest.approx(1.0)

    def test_ep0_skips_causal_factor(self) -> None:
        fb = (
            "Implement a REST API for user management"
            " with CRUD operations"
        )
        score = score_episode_quality(
            feedback_body=fb,
            action_diff="+class UserController:\n+    pass",
            is_ep0=True,
        )
        cfg = QualityWeightConfig()
        assert score == pytest.approx(cfg.feedback_moderate_factor)
        non_ep0 = score_episode_quality(
            feedback_body=fb,
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
# compute_causal_density
# ---------------------------------------------------------------------------


class TestComputeCausalDensity:
    def test_high_overlap(self) -> None:
        fb = "rename parse_response to handle_response"
        code = "+def handle_response(data):\n+    return parse_response(data)"
        density = compute_causal_density(fb, code)
        # "rename" is in fb but not code → 2/3
        assert density == pytest.approx(2.0 / 3.0)

    def test_partial_overlap(self) -> None:
        fb = "the validate_input function should check for None"
        code = "def validate_input(s):\n    if s is None: return ''"
        density = compute_causal_density(fb, code)
        assert 0.0 < density < 1.0

    def test_no_overlap(self) -> None:
        fb = "this approach is fragile and should be reworked"
        code = "+def validate(x):\n+    if x < 0: raise ValueError"
        density = compute_causal_density(fb, code)
        assert density == 0.0

    def test_url_only_feedback(self) -> None:
        density = compute_causal_density("https://example.com", "def foo(): pass")
        assert density == 0.0

    def test_empty_feedback(self) -> None:
        assert compute_causal_density("", "def foo(): pass") == 0.0


# ---------------------------------------------------------------------------
# compute_diff_focus
# ---------------------------------------------------------------------------


class TestComputeDiffFocus:
    def test_identical(self) -> None:
        code = "def foo():\n    return 42"
        assert compute_diff_focus(code, code) == pytest.approx(1.0)

    def test_focused_change(self) -> None:
        before = "def foo(x):\n    return x + 1\n\ndef bar():\n    return 0"
        after = "def foo(x):\n    return x + 2\n\ndef bar():\n    return 0"
        sim = compute_diff_focus(before, after)
        assert 0.85 <= sim < 1.0

    def test_major_rewrite(self) -> None:
        before = "def foo():\n    return 1"
        after = "class Bar:\n    def baz(self) -> int:\n        return 42"
        sim = compute_diff_focus(before, after)
        assert sim < 0.70

    def test_empty_before(self) -> None:
        assert compute_diff_focus("", "def foo(): pass") == 0.0

    def test_empty_after(self) -> None:
        assert compute_diff_focus("def foo(): pass", "") == 0.0


# ---------------------------------------------------------------------------
# score_external_quality
# ---------------------------------------------------------------------------


class TestScoreExternalQuality:
    def test_high_quality_external(self) -> None:
        fb = (
            "The validate_input function should check for None"
            " before accessing .strip() to avoid AttributeError"
        )
        before = (
            "def validate_input(s):\n    return s.strip()"
        )
        after = (
            "def validate_input(s):\n"
            "    if s is None:\n        return ''\n"
            "    return s.strip()"
        )
        score = score_external_quality(
            feedback_body=fb, before_code=before, after_code=after,
        )
        assert score > 0.5

    def test_best_case_external_can_reach_one(self) -> None:
        fb = (
            "The parse_response function should validate the"
            " status_code field before accessing the data"
            " payload to prevent KeyError"
        )
        unchanged = "\n".join(
            f"    line_{i} = {i}" for i in range(20)
        )
        before = (
            f'def parse_response(response):\n'
            f'    """Parse API response."""\n'
            f'{unchanged}\n'
            f'    return response["data"]'
        )
        after = (
            f'def parse_response(response):\n'
            f'    """Parse API response."""\n'
            f'{unchanged}\n'
            f'    if response.get("status_code") != 200:\n'
            f'        return None\n'
            f'    return response["data"]'
        )
        score = score_external_quality(
            feedback_body=fb, before_code=before, after_code=after,
        )
        assert score == pytest.approx(1.0)

    def test_no_overlap_vague_feedback_scores_low(self) -> None:
        score = score_external_quality(
            feedback_body="Really good idea to allow them to be injected!",
            before_code="x = 1\ny = 2\nz = 3",
            after_code="a = 10\nb = 20\nc = 30",
        )
        assert score < 0.15

    def test_short_feedback_large_diff_penalized(self) -> None:
        score = score_external_quality(
            feedback_body="fix",
            before_code="x",
            after_code="+" * 6000,
        )
        cfg = QualityWeightConfig()
        assert score <= cfg.proportionality_penalty

    def test_floor_respected(self) -> None:
        score = score_external_quality(
            feedback_body="https://example.com",
            before_code="x",
            after_code="+" * 6000,
        )
        assert score >= QualityWeightConfig().floor

    def test_trivial_diff_penalized(self) -> None:
        code = "def foo():\n    return 42\n" * 20
        after = code[:-1] + " "  # near-identical
        fb = (
            "The foo function should be cleaned up and improved"
            " for readability and consistency with the"
            " codebase style"
        )
        score = score_external_quality(
            feedback_body=fb,
            before_code=code,
            after_code=after,
        )
        cfg = QualityWeightConfig()
        assert score <= cfg.diff_focus_trivial_factor
