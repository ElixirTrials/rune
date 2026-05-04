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
    assert "d0" in pairs[0]["teacher_text"]
    assert "d1" in pairs[1]["teacher_text"]
    assert "d2" in pairs[2]["teacher_text"]
    for p in pairs:
        assert "## Revision\n" in p["teacher_text"]


def test_unroll_prompt_contains_prior_diff_and_feedback() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    a1 = pairs[1]["activation_text"]
    assert "d0" in a1  # prior diff visible
    assert "rename foo" in a1  # feedback body visible
    assert "## Current Code\n" in a1
    a2 = pairs[2]["activation_text"]
    assert "tests/test_foo.py::test_bar" in a2  # summary visible


def test_unroll_carries_task_id_and_metadata() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    assert pairs[0]["metadata"]["source_task_id"] == "pr_owner/repo_1"
    assert pairs[0]["metadata"]["step_index"] == 0
    assert pairs[2]["metadata"]["step_index"] == 2


def test_unroll_has_pre_post_code_side_channels() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    assert pairs[0]["pre_code"] == ""
    assert pairs[0]["post_code"] == "d0"
    assert pairs[1]["pre_code"] == "d0"
    assert pairs[1]["post_code"] == "d1"


def test_unroll_activation_text_teacher_text_schema() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    for p in pairs:
        assert "activation_text" in p
        assert "teacher_text" in p
        assert p["teacher_text"].startswith(p["activation_text"])


def test_unroll_has_quality_score() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    for p in pairs:
        assert "quality_score" in p
        assert 0.0 < p["quality_score"] <= 1.0
        assert p["metadata"]["quality_score"] == p["quality_score"]


def test_unroll_ep0_not_penalized_by_causal() -> None:
    pairs = unroll_trajectory_to_pairs(_traj_3_rounds())
    ep0_score = pairs[0]["quality_score"]
    # ep0 body is "goal" (4 chars = short feedback), source=1.0, causal=1.0
    # Non-ep0 with same short body and no overlap would get causal=0.4
    assert ep0_score > 0.2


def test_unroll_url_only_feedback_hits_floor() -> None:
    from model_training.d2l_models import Episode, Feedback, FeedbackKind

    traj = _traj_3_rounds()
    traj.episodes[1] = Episode(
        round=1,
        prior_diff="d0",
        feedback=Feedback(
            kind=FeedbackKind.ci_failure,
            body="https://circleci.com/gh/org/repo/123",
        ),
        action_diff="d1",
    )
    pairs = unroll_trajectory_to_pairs(traj)
    assert pairs[1]["quality_score"] == 0.05
