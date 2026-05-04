"""Tests for feedback↔next-commit pairing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from model_training.d2l_pairing import FeedbackEvent, pair_feedback_with_commits

T0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _commit(sha: str, minutes: int, author: str) -> dict:
    return {
        "sha": sha,
        "commit": {"committer": {"date": (T0 + timedelta(minutes=minutes)).isoformat()}},
        "author": {"login": author},
    }


def _fb(kind: str, minutes: int, body: str) -> FeedbackEvent:
    return FeedbackEvent(
        kind=kind,
        body=body,
        ts=T0 + timedelta(minutes=minutes),
        author="reviewer",
        anchor=None,
    )


def test_review_comment_pairs_with_next_author_commit() -> None:
    commits = [_commit("c0", 0, "alice"), _commit("c1", 30, "alice")]
    feedback = [_fb("review_comment", 10, "rename foo")]
    rounds = pair_feedback_with_commits(commits, feedback, pr_author="alice")
    assert len(rounds) == 1
    assert rounds[0].next_commit["sha"] == "c1"
    assert rounds[0].feedback.body == "rename foo"


def test_review_comment_after_last_commit_is_dropped() -> None:
    commits = [_commit("c0", 0, "alice")]
    feedback = [_fb("review_comment", 10, "but no fix landed")]
    rounds = pair_feedback_with_commits(commits, feedback, pr_author="alice")
    assert rounds == []


def test_ci_failure_pairs_with_next_commit_by_anyone() -> None:
    commits = [_commit("c0", 0, "alice"), _commit("c1", 5, "bob")]
    feedback = [_fb("ci_failure", 2, "test x failed")]
    rounds = pair_feedback_with_commits(commits, feedback, pr_author="alice")
    assert len(rounds) == 1
    assert rounds[0].next_commit["sha"] == "c1"


def test_multiple_rounds_chronologically_ordered() -> None:
    commits = [
        _commit("c0", 0, "alice"),
        _commit("c1", 30, "alice"),
        _commit("c2", 60, "alice"),
    ]
    feedback = [
        _fb("review_comment", 10, "first"),
        _fb("ci_failure", 40, "second"),
    ]
    rounds = pair_feedback_with_commits(commits, feedback, pr_author="alice")
    assert [r.next_commit["sha"] for r in rounds] == ["c1", "c2"]
    assert [r.feedback.body for r in rounds] == ["first", "second"]
