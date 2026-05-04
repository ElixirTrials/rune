"""Tests for the corrective-richness quality filter (paper Gate 1 + 3 setup)."""

from __future__ import annotations

from model_training.d2l_mining import score_pr_quality


def test_score_rewards_anchored_review_comments() -> None:
    pr = {
        "user": {"login": "alice"},
        "review_comments_with_anchor": 3,
        "n_commits": 5,
        "ci_failures_resolved": 1,
        "labels": [],
        "n_files_changed_per_commit_p95": 8,
        "merged_at": "2026-04-01T00:00:00Z",
    }
    assert score_pr_quality(pr) > 0


def test_score_excludes_bot_authors() -> None:
    pr = {
        "user": {"login": "dependabot[bot]"},
        "review_comments_with_anchor": 5,
        "n_commits": 4,
        "ci_failures_resolved": 0,
        "labels": [],
        "n_files_changed_per_commit_p95": 4,
        "merged_at": "2026-04-01T00:00:00Z",
    }
    assert score_pr_quality(pr) == 0


def test_score_excludes_doc_only_labels() -> None:
    pr = {
        "user": {"login": "alice"},
        "review_comments_with_anchor": 5,
        "n_commits": 4,
        "ci_failures_resolved": 0,
        "labels": [{"name": "documentation"}],
        "n_files_changed_per_commit_p95": 4,
        "merged_at": "2026-04-01T00:00:00Z",
    }
    assert score_pr_quality(pr) == 0


def test_score_excludes_unmerged() -> None:
    pr = {
        "user": {"login": "alice"},
        "review_comments_with_anchor": 5,
        "n_commits": 4,
        "ci_failures_resolved": 0,
        "labels": [],
        "n_files_changed_per_commit_p95": 4,
        "merged_at": None,
    }
    assert score_pr_quality(pr) == 0


def test_score_penalises_mass_edit_commits() -> None:
    base = {
        "user": {"login": "alice"},
        "review_comments_with_anchor": 2,
        "n_commits": 4,
        "ci_failures_resolved": 1,
        "labels": [],
        "merged_at": "2026-04-01T00:00:00Z",
    }
    small = score_pr_quality({**base, "n_files_changed_per_commit_p95": 5})
    huge = score_pr_quality({**base, "n_files_changed_per_commit_p95": 100})
    assert small > huge
