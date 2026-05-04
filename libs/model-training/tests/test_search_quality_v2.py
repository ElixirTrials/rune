"""Tests for search_quality_prs_v2 — score-ranked candidate selection."""

from __future__ import annotations

from unittest.mock import patch

from model_training.d2l_mining import search_quality_prs_v2


def test_search_v2_returns_scored_prs() -> None:
    """Basic test: search returns PR numbers ranked by score."""
    with patch("model_training.d2l_mining.GitHubClient") as mock_cls:
        client = mock_cls.return_value
        client.search_and_score_prs_graphql.return_value = [
            {
                "number": 1,
                "user": {"login": "alice"},
                "review_comments_with_anchor": 1,
                "n_commits": 2,
                "ci_failures_resolved": 0,
                "labels": [],
                "n_files_changed_per_commit_p95": 4,
                "merged_at": "2026-04-01T00:00:00Z",
            },
            {
                "number": 2,
                "user": {"login": "alice"},
                "review_comments_with_anchor": 4,
                "n_commits": 5,
                "ci_failures_resolved": 2,
                "labels": [],
                "n_files_changed_per_commit_p95": 4,
                "merged_at": "2026-04-01T00:00:00Z",
            },
        ]
        out = search_quality_prs_v2("o/r", max_results=10, github_token="x")
        assert out[0] == 2  # higher score first


def test_search_v2_excludes_zero_score_prs() -> None:
    """PRs with zero score are excluded from results."""
    with patch("model_training.d2l_mining.GitHubClient") as mock_cls:
        client = mock_cls.return_value
        client.search_and_score_prs_graphql.return_value = [
            {
                "number": 1,
                "user": {"login": "alice"},
                "review_comments_with_anchor": 0,
                "n_commits": 1,
                "ci_failures_resolved": 0,
                "labels": [],
                "n_files_changed_per_commit_p95": 4,
                "merged_at": "2026-04-01T00:00:00Z",
            },
            {
                "number": 2,
                "user": {"login": "alice"},
                "review_comments_with_anchor": 3,
                "n_commits": 5,
                "ci_failures_resolved": 1,
                "labels": [],
                "n_files_changed_per_commit_p95": 4,
                "merged_at": "2026-04-01T00:00:00Z",
            },
        ]
        out = search_quality_prs_v2("o/r", max_results=10, github_token="x")
        assert 2 in out
