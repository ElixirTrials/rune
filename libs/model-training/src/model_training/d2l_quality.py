"""Per-datapoint quality heuristic for SFT loss weighting.

Computes a composite quality_score in [0.05, 1.0] that modulates gradient
signal via multiplication into per-token hunk weights.  All scoring is
deterministic, regex-based, and requires no LLM or network access.

The score decomposes multiplicatively::

    quality_score = max(FLOOR, source * causal * feedback * proportionality)

Each factor and its thresholds are exposed via QualityWeightConfig so
callers can tune without touching this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CausalLink",
    "DataSource",
    "QualityWeightConfig",
    "classify_causal_link",
    "score_episode_quality",
    "score_external_quality",
]

DataSource = Literal["trajectory", "external_single_turn"]
CausalLink = Literal["entity_overlap", "no_overlap", "url_only"]

_URL_RE = re.compile(r"https?://\S+")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


@dataclass
class QualityWeightConfig:
    """Configurable weights for the composite quality score."""

    source_trajectory: float = 1.0
    source_external: float = 0.4

    causal_entity_overlap: float = 1.0
    causal_no_overlap: float = 0.4
    causal_url_only: float = 0.05

    feedback_rich_chars: int = 100
    feedback_moderate_chars: int = 20
    feedback_rich_factor: float = 1.0
    feedback_moderate_factor: float = 0.7
    feedback_short_factor: float = 0.4
    feedback_url_only_factor: float = 0.1

    proportionality_short_chars: int = 20
    proportionality_diff_chars: int = 5000
    proportionality_penalty: float = 0.3

    floor: float = 0.05


def is_url_only(text: str) -> bool:
    """Return True when the text contains only URLs and whitespace."""
    stripped = text.strip()
    if not stripped:
        return False
    return all(_URL_RE.match(w) for w in stripped.split())


def classify_causal_link(feedback_body: str, action_diff: str) -> CausalLink:
    """Classify the causal relationship between feedback and action_diff.

    Extracts identifiers (``[A-Za-z_][A-Za-z0-9_]{2,}``) from both texts
    and checks for set intersection.  URL-only feedback is classified
    separately since it carries no actionable signal regardless of overlap.

    Args:
        feedback_body: Raw feedback text.
        action_diff: The code change the episode produced.

    Returns:
        One of ``"entity_overlap"``, ``"no_overlap"``, or ``"url_only"``.
    """
    if is_url_only(feedback_body):
        return "url_only"
    fb_ids = set(_IDENTIFIER_RE.findall(feedback_body))
    diff_ids = set(_IDENTIFIER_RE.findall(action_diff))
    if fb_ids & diff_ids:
        return "entity_overlap"
    return "no_overlap"


def _classify_feedback(text: str, config: QualityWeightConfig) -> float:
    stripped = text.strip()
    if not stripped:
        return config.feedback_short_factor
    if is_url_only(stripped):
        return config.feedback_url_only_factor
    n = len(stripped)
    if n >= config.feedback_rich_chars:
        return config.feedback_rich_factor
    if n >= config.feedback_moderate_chars:
        return config.feedback_moderate_factor
    return config.feedback_short_factor


def score_episode_quality(
    *,
    feedback_body: str,
    action_diff: str,
    is_ep0: bool,
    config: QualityWeightConfig | None = None,
) -> float:
    """Compute quality_score for one trajectory episode.

    Args:
        feedback_body: Raw feedback text (body field of Feedback model).
        action_diff: The code change the episode produced.
        is_ep0: True for the task_description episode (round==0).
        config: Quality factor configuration.  Uses defaults when None.

    Returns:
        Float in [config.floor, 1.0].
    """
    cfg = config or QualityWeightConfig()

    source_factor = cfg.source_trajectory

    if is_ep0:
        causal_factor = 1.0
    else:
        link = classify_causal_link(feedback_body, action_diff)
        if link == "entity_overlap":
            causal_factor = cfg.causal_entity_overlap
        elif link == "no_overlap":
            causal_factor = cfg.causal_no_overlap
        else:
            causal_factor = cfg.causal_url_only

    feedback_factor = _classify_feedback(feedback_body, cfg)

    proportionality_factor = 1.0
    if (
        len(feedback_body.strip()) < cfg.proportionality_short_chars
        and len(action_diff) > cfg.proportionality_diff_chars
    ):
        proportionality_factor = cfg.proportionality_penalty

    raw = source_factor * causal_factor * feedback_factor * proportionality_factor
    return max(cfg.floor, min(raw, 1.0))


def score_external_quality(
    *,
    feedback_body: str,
    before_code: str,
    after_code: str,
    config: QualityWeightConfig | None = None,
) -> float:
    """Compute quality_score for one external single-turn record.

    Args:
        feedback_body: The reviewer comment text.
        before_code: Code snippet before the review.
        after_code: Code snippet after the review.
        config: Quality factor configuration.  Uses defaults when None.

    Returns:
        Float in [config.floor, 1.0].
    """
    cfg = config or QualityWeightConfig()

    source_factor = cfg.source_external
    feedback_factor = _classify_feedback(feedback_body, cfg)

    proportionality_factor = 1.0
    if (
        len(feedback_body.strip()) < cfg.proportionality_short_chars
        and len(after_code) > cfg.proportionality_diff_chars
    ):
        proportionality_factor = cfg.proportionality_penalty

    raw = source_factor * feedback_factor * proportionality_factor
    return max(cfg.floor, min(raw, 1.0))
