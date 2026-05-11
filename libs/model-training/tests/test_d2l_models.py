"""Tests for trajectory-mining Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from model_training.d2l_models import (
    Anchor,
    Episode,
    Feedback,
    FeedbackKind,
    Provenance,
    Trajectory,
)
from pydantic import ValidationError


def _provenance() -> Provenance:
    return Provenance(
        repo="owner/repo",
        pr_number=42,
        license="MIT",
        head_sha="a" * 40,
        base_sha="b" * 40,
        mined_at=datetime(2026, 5, 3, tzinfo=timezone.utc),
    )


def test_feedback_review_comment_with_anchor() -> None:
    fb = Feedback(
        kind=FeedbackKind.review_comment,
        body="this allocates inside the hot loop — pull it out",
        author="reviewer1",
        anchor=Anchor(file="src/foo.py", line=42),
    )
    assert fb.kind == FeedbackKind.review_comment
    assert fb.summary is None
    assert fb.anchor.line == 42


def test_feedback_test_failure_with_summary() -> None:
    fb = Feedback(
        kind=FeedbackKind.test_failure,
        body="full pytest output goes here",
        summary="tests/test_foo.py::test_bar - AssertionError",
        anchor=Anchor(test="tests/test_foo.py::test_bar"),
    )
    assert fb.summary.startswith("tests/test_foo.py")


def test_episode_round_zero_has_empty_prior_diff() -> None:
    ep = Episode(
        round=0,
        prior_diff="",
        feedback=Feedback(kind=FeedbackKind.task_description, body="goal"),
        action_diff="--- foo.py ---\n@@ -1,2 +1,3 @@\n+x = 1\n",
    )
    assert ep.round == 0
    assert ep.prior_diff == ""


def test_trajectory_roundtrip() -> None:
    traj = Trajectory(
        task_id="pr_owner/repo_42",
        task_description="Add feature X",
        episodes=[
            Episode(
                round=0,
                prior_diff="",
                feedback=Feedback(kind=FeedbackKind.task_description, body="goal"),
                action_diff="diff0",
            ),
            Episode(
                round=1,
                prior_diff="diff0",
                feedback=Feedback(
                    kind=FeedbackKind.review_comment,
                    body="rename foo",
                    author="rev",
                ),
                action_diff="diff1",
            ),
        ],
        metadata={"outcome": "merged", "language": "python", "n_rounds": 2},
        provenance=_provenance(),
    )
    raw = traj.model_dump_json()
    parsed = Trajectory.model_validate_json(raw)
    assert parsed.episodes[1].feedback.kind == FeedbackKind.review_comment


def test_provenance_rejects_short_sha() -> None:
    with pytest.raises(ValidationError):
        Provenance(
            repo="owner/repo",
            pr_number=1,
            license="MIT",
            head_sha="abc",  # too short
            base_sha="b" * 40,
            mined_at=datetime(2026, 5, 3, tzinfo=timezone.utc),
        )
