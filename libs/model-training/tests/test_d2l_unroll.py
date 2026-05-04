"""Tests for trajectory→pairs unrolling (Gate 1 dual-shape requirement)."""

from __future__ import annotations

from datetime import datetime, timezone

from model_training.d2l_data import unroll_trajectory_to_pairs
from model_training.d2l_models import (
    Anchor,
    Episode,
    Feedback,
    FeedbackKind,
    Provenance,
    Trajectory,
)


def _traj_3_rounds() -> Trajectory:
    prov = Provenance(
        repo="owner/repo",
        pr_number=1,
        license="MIT",
        head_sha="a" * 40,
        base_sha="b" * 40,
        mined_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    eps = [
        Episode(
            round=0,
            prior_diff="",
            feedback=Feedback(kind=FeedbackKind.task_description, body="goal"),
            action_diff="d0",
        ),
        Episode(
            round=1,
            prior_diff="d0",
            feedback=Feedback(
                kind=FeedbackKind.review_comment,
                body="rename foo",
                author="rev",
                anchor=Anchor(file="src/foo.py", line=1),
            ),
            action_diff="d1",
        ),
        Episode(
            round=2,
            prior_diff="d0\nd1",
            feedback=Feedback(
                kind=FeedbackKind.test_failure,
                body="full pytest output",
                summary="tests/test_foo.py::test_bar - AssertionError",
            ),
            action_diff="d2",
        ),
    ]
    return Trajectory(
        task_id="pr_owner/repo_1",
        task_description="goal",
        episodes=eps,
        metadata={"outcome": "merged"},
        provenance=prov,
    )


def test_unroll_emits_one_pair_per_episode() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    assert len(pairs) == 3


def test_unroll_target_is_action_diff() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    assert pairs[0]["response"] == "d0"
    assert pairs[1]["response"] == "d1"
    assert pairs[2]["response"] == "d2"


def test_unroll_prompt_contains_prior_diff_and_feedback() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    p1 = pairs[1]["prompt"]
    assert "d0" in p1  # prior diff visible
    assert "rename foo" in p1  # feedback body visible
    p2 = pairs[2]["prompt"]
    assert "tests/test_foo.py::test_bar" in p2  # summary visible


def test_unroll_carries_task_id_and_round() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    assert pairs[0]["task_id"] == "pr_owner/repo_1"
    assert pairs[0]["round"] == 0
    assert pairs[2]["round"] == 2
