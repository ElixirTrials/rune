"""Tests for mine_pr_diff_chains and mine_issue_commit_chains."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from model_training.d2l_mining import (
    mine_issue_commit_chains,
    mine_pr_diff_chains,
    search_quality_prs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pr(
    number: int = 1,
    title: str = "Fix the widget",
    body: str = "This PR fixes the broken widget.",
    state: str = "closed",
    merged: bool = True,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "merged_at": "2026-01-01T00:00:00Z" if merged else None,
    }


def _make_commit(
    sha: str, message: str, date: str = "2026-01-01T00:00:00Z"
) -> dict[str, Any]:
    return {
        "sha": sha,
        "commit": {"message": message, "committer": {"date": date}},
    }


def _make_commit_detail(sha: str, patch: str) -> dict[str, Any]:
    return {"sha": sha, "files": [{"filename": "widget.py", "patch": patch}]}


def _make_review_comment(
    body: str, created_at: str = "2026-01-01T01:00:00Z"
) -> dict[str, Any]:
    return {"body": body, "created_at": created_at}


def _make_issue(
    number: int = 1,
    title: str = "Bug in parser",
    body: str = "The parser crashes on empty input.",
    state: str = "closed",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "pull_request": None,
    }


def _make_repo_commit(sha: str, message: str) -> dict[str, Any]:
    return {"sha": sha, "commit": {"message": message}}


# ---------------------------------------------------------------------------
# TestMinePrDiffChains
# ---------------------------------------------------------------------------


@patch("model_training.d2l_mining.GitHubClient")
class TestMinePrDiffChains:
    """Tests for mine_pr_diff_chains."""

    def test_returns_trajectory_with_required_fields(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_pr(number=1, merged=True)],  # PRs
            [_make_commit("abc123", "initial commit")],  # commits for PR 1
            [_make_review_comment("Looks good")],  # reviews for PR 1
        ]
        client.get.return_value = _make_commit_detail("abc123", "+print('hi')")

        result = mine_pr_diff_chains("owner/repo", max_prs=10)

        assert len(result) == 1
        traj = result[0]
        assert traj["task_id"] == "pr_owner/repo_1"
        assert "Fix the widget" in traj["task_description"]
        assert "fixes the broken widget" in traj["task_description"]
        assert isinstance(traj["steps"], list)
        assert len(traj["steps"]) > 0
        assert traj["outcome"] == "merged"

    def test_closed_unmerged_pr_has_closed_outcome(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_pr(number=2, merged=False)],
            [_make_commit("def456", "wip")],
            [],  # no reviews
        ]
        client.get.return_value = _make_commit_detail("def456", "+x = 1")

        result = mine_pr_diff_chains("owner/repo")

        assert result[0]["outcome"] == "closed"

    def test_steps_contain_commit_diffs(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_pr(number=3)],
            [
                _make_commit("aaa", "first change"),
                _make_commit("bbb", "second change"),
            ],
            [],  # no reviews
        ]
        client.get.side_effect = [
            _make_commit_detail("aaa", "+line1"),
            _make_commit_detail("bbb", "+line2"),
        ]

        result = mine_pr_diff_chains("owner/repo")

        commit_steps = [s for s in result[0]["steps"] if s["type"] == "commit"]
        assert len(commit_steps) == 2
        assert "+line1" in commit_steps[0]["content"]
        assert "+line2" in commit_steps[1]["content"]

    def test_steps_include_review_comments(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_pr(number=4)],
            [_make_commit("ccc", "code change", date="2026-01-01T02:00:00Z")],
            [_make_review_comment("Please add tests", "2026-01-01T01:00:00Z")],
        ]
        client.get.return_value = _make_commit_detail("ccc", "+impl")

        result = mine_pr_diff_chains("owner/repo")

        steps = result[0]["steps"]
        # Review (01:00) should come BEFORE commit (02:00) chronologically
        assert steps[0]["type"] == "review"
        assert steps[0]["content"] == "Please add tests"
        assert steps[1]["type"] == "commit"

    def test_empty_repo_returns_empty_list(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [],  # no PRs
        ]

        result = mine_pr_diff_chains("owner/repo")

        assert result == []

    def test_max_prs_limits_results(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        prs = [_make_pr(number=i) for i in range(1, 6)]

        # side_effect: first call returns all 5 PRs, then per-PR calls
        side_effects: list[Any] = [prs]
        for i in range(1, 6):
            side_effects.append([_make_commit(f"sha{i}", f"commit {i}")])
            side_effects.append([])  # no reviews
        client.get_paginated.side_effect = side_effects
        client.get.side_effect = [
            _make_commit_detail(f"sha{i}", f"+code{i}") for i in range(1, 6)
        ]

        result = mine_pr_diff_chains("owner/repo", max_prs=2)

        assert len(result) <= 2


# ---------------------------------------------------------------------------
# TestMineIssueCommitChains
# ---------------------------------------------------------------------------


@patch("model_training.d2l_mining.GitHubClient")
class TestMineIssueCommitChains:
    """Tests for mine_issue_commit_chains."""

    def test_returns_trajectory_with_required_fields(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_issue(number=7, state="closed")],  # issues
            [_make_repo_commit("abc", "fixes #7")],  # repo commits
        ]
        client.get.return_value = _make_commit_detail("abc", "+fix")

        result = mine_issue_commit_chains("owner/repo")

        assert len(result) == 1
        traj = result[0]
        assert traj["task_id"] == "issue_owner/repo_7"
        assert "Bug in parser" in traj["task_description"]
        assert "crashes on empty input" in traj["task_description"]
        assert isinstance(traj["steps"], list)
        assert len(traj["steps"]) > 0
        assert traj["outcome"] == "closed"

    def test_matches_fixes_closes_resolves_patterns(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_issue(number=5)],
            [
                _make_repo_commit("s1", "fixes #5"),
                _make_repo_commit("s2", "Closes #5"),
                _make_repo_commit("s3", "resolves #5"),
            ],
        ]
        client.get.side_effect = [
            _make_commit_detail("s1", "+a"),
            _make_commit_detail("s2", "+b"),
            _make_commit_detail("s3", "+c"),
        ]

        result = mine_issue_commit_chains("owner/repo")

        assert len(result) == 1
        assert len(result[0]["steps"]) == 3

    def test_open_issue_has_open_outcome(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_issue(number=9, state="open")],
            [_make_repo_commit("xyz", "fixes #9")],
        ]
        client.get.return_value = _make_commit_detail("xyz", "+wip")

        result = mine_issue_commit_chains("owner/repo")

        assert result[0]["outcome"] == "open"

    def test_issue_with_no_linked_commits_is_skipped(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get_paginated.side_effect = [
            [_make_issue(number=10)],
            [_make_repo_commit("zzz", "unrelated commit")],  # no ref to #10
        ]

        result = mine_issue_commit_chains("owner/repo")

        assert result == []

    def test_pull_request_issues_are_excluded(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        pr_item: dict[str, Any] = {
            "number": 20,
            "title": "A PR not an issue",
            "body": "PR body",
            "state": "closed",
            "pull_request": {"url": "https://api.github.com/..."},
        }
        real_issue = _make_issue(number=21)
        client.get_paginated.side_effect = [
            [pr_item, real_issue],  # issues endpoint returns both
            [_make_repo_commit("mmm", "fixes #21")],
        ]
        client.get.return_value = _make_commit_detail("mmm", "+ok")

        result = mine_issue_commit_chains("owner/repo")

        assert len(result) == 1
        assert result[0]["task_id"] == "issue_owner/repo_21"


# ---------------------------------------------------------------------------
# Helpers for search_quality_prs
# ---------------------------------------------------------------------------


def _make_search_item(
    number: int, labels: list[str] | None = None, comments: int = 3
) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"PR #{number}",
        "body": "description",
        "state": "closed",
        "pull_request": {"merged_at": "2026-01-01T00:00:00Z"},
        "labels": [{"name": lbl} for lbl in (labels or [])],
        "comments": comments,
    }


# ---------------------------------------------------------------------------
# TestSearchQualityPrs
# ---------------------------------------------------------------------------


@patch("model_training.d2l_mining.GitHubClient")
class TestSearchQualityPrs:
    """Tests for search_quality_prs."""

    def test_search_returns_merged_prs_only(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get.side_effect = [
            # search results
            {
                "total_count": 2,
                "items": [_make_search_item(10), _make_search_item(20)],
            },
            # PR detail for #10
            {"number": 10, "commits": 3},
            # PR detail for #20
            {"number": 20, "commits": 3},
        ]

        result = search_quality_prs("owner/repo", max_results=10)

        assert result == [10, 20]

    def test_excludes_labeled_prs(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get.side_effect = [
            {
                "total_count": 2,
                "items": [
                    _make_search_item(10, labels=["dependencies"]),
                    _make_search_item(20, labels=["feature"]),
                ],
            },
            # Only PR #20 passes label filter, so only its detail is fetched
            {"number": 20, "commits": 2},
        ]

        result = search_quality_prs("owner/repo", max_results=10)

        assert result == [20]

    def test_min_commits_filter(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get.side_effect = [
            {
                "total_count": 2,
                "items": [_make_search_item(10), _make_search_item(20)],
            },
            {"number": 10, "commits": 1},
            {"number": 20, "commits": 3},
        ]

        result = search_quality_prs("owner/repo", min_commits=2)

        assert result == [20]

    def test_empty_search_returns_empty(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        client.get.return_value = {"total_count": 0, "items": []}

        result = search_quality_prs("owner/repo")

        assert result == []

    def test_mine_with_pr_numbers(self, mock_cls: Any) -> None:
        client = mock_cls.return_value
        pr_data = _make_pr(number=42, title="Quality PR", body="Good stuff")
        client.get.side_effect = [
            pr_data,  # individual PR fetch
            _make_commit_detail("sha42", "+quality"),  # commit detail
        ]
        client.get_paginated.side_effect = [
            [_make_commit("sha42", "quality commit")],  # commits
            [_make_review_comment("LGTM")],  # reviews
        ]

        result = mine_pr_diff_chains("owner/repo", pr_numbers=[42])

        # Verify get_paginated was NOT called with the pulls list endpoint
        for call in client.get_paginated.call_args_list:
            path = call[0][0]
            assert path != "/repos/owner/repo/pulls" or (
                "commits" in path or "comments" in path
            )
        assert len(result) == 1
        assert result[0]["task_id"] == "pr_owner/repo_42"
