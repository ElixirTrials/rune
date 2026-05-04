"""Tests for search_quality_prs_v2 — score-ranked candidate selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from model_training.d2l_mining import search_quality_prs_v2


def test_search_v2_returns_scored_prs() -> None:
    """Basic test: search returns PR numbers ranked by score."""
    with patch("model_training.d2l_mining.GitHubClient") as Client:
        client = Client.return_value
        client.get.return_value = {
            "items": [
                {"number": 1, "labels": []},
                {"number": 2, "labels": []},
            ]
        }

        def fake_features(c, repo, pr_num):
            if pr_num == 1:
                return {
                    "user": {"login": "alice"},
                    "review_comments_with_anchor": 1,
                    "n_commits": 2,
                    "ci_failures_resolved": 0,
                    "labels": [],
                    "n_files_changed_per_commit_p95": 4,
                    "merged_at": "2026-04-01T00:00:00Z",
                }
            return {
                "user": {"login": "alice"},
                "review_comments_with_anchor": 4,
                "n_commits": 5,
                "ci_failures_resolved": 2,
                "labels": [],
                "n_files_changed_per_commit_p95": 4,
                "merged_at": "2026-04-01T00:00:00Z",
            }

        with patch("model_training.d2l_mining._features_for_pr", side_effect=fake_features):
            out = search_quality_prs_v2("o/r", max_results=10, github_token="x")
            assert out[0] == 2  # higher score first


def test_search_v2_excludes_doc_labels() -> None:
    """PRs with documentation labels are pre-filtered before scoring."""
    with patch("model_training.d2l_mining.GitHubClient") as Client:
        client = Client.return_value
        client.get.return_value = {
            "items": [
                {"number": 1, "labels": [{"name": "documentation"}]},
                {"number": 2, "labels": []},
            ]
        }

        def fake_features(c, repo, pr_num):
            return {
                "user": {"login": "alice"},
                "review_comments_with_anchor": 3,
                "n_commits": 5,
                "ci_failures_resolved": 1,
                "labels": [],
                "n_files_changed_per_commit_p95": 4,
                "merged_at": "2026-04-01T00:00:00Z",
            }

        with patch("model_training.d2l_mining._features_for_pr", side_effect=fake_features):
            out = search_quality_prs_v2("o/r", max_results=10, github_token="x")
            assert out == [2]  # PR 1 excluded by label
