"""Integration test for mine_pr_trajectories with a stubbed GitHubClient."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from model_training.d2l_mining import mine_pr_trajectories
from model_training.d2l_models import FeedbackKind

T0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _commit(sha: str, minutes: int, author: str, files: list[dict]) -> dict:
    return {
        "sha": sha,
        "commit": {
            "message": f"commit {sha}",
            "committer": {"date": (T0 + timedelta(minutes=minutes)).isoformat()},
        },
        "author": {"login": author},
        "files": files,
    }


@pytest.fixture
def fake_client() -> MagicMock:
    client = MagicMock()
    client.get_repo_license.return_value = "MIT"

    def get(path: str) -> dict:
        if path == "/repos/owner/repo/pulls/1":
            return {
                "number": 1,
                "title": "Add feature X",
                "body": "Implements X",
                "user": {"login": "alice"},
                "merged_at": "2026-05-02T00:00:00Z",
                "head": {"sha": "a" * 40},
                "base": {"sha": "b" * 40},
                "labels": [],
            }
        if path.endswith("/check-runs"):
            return {"check_runs": []}
        if path.endswith("/check-suites"):
            return {"check_suites": []}
        sha = path.rsplit("/", 1)[-1]
        return {
            "files": [{"filename": "src/foo.py", "patch": f"@@ -1,1 +1,2 @@\n+{sha}\n"}]
        }

    client.get.side_effect = get

    def paginated(path: str, **_kwargs):
        if path.endswith("/commits"):
            return [
                _commit("c0", 0, "alice",
                        [{"filename": "src/foo.py", "patch": "@@ -1,1 +1,2 @@\n+x\n"}]),
                _commit("c1", 30, "alice",
                        [{"filename": "src/foo.py", "patch": "@@ -1,1 +1,2 @@\n+y\n"}]),
            ]
        if path.endswith("/comments"):
            return [
                {
                    "user": {"login": "rev"},
                    "body": "rename foo",
                    "created_at": (T0 + timedelta(minutes=10)).isoformat(),
                    "path": "src/foo.py",
                    "line": 1,
                },
            ]
        if "/issues/" in path:
            return []
        return []

    client.get_paginated.side_effect = paginated
    client.get_check_runs.return_value = []
    return client


def test_mine_pr_trajectories_yields_trajectory_with_two_episodes(fake_client) -> None:
    out = mine_pr_trajectories(
        "owner/repo",
        pr_numbers=[1],
        github_client=fake_client,
    )
    assert len(out) == 1
    traj = out[0]
    assert traj.task_id == "pr_owner/repo_1"
    assert len(traj.episodes) == 2
    assert traj.episodes[0].round == 0
    assert traj.episodes[0].feedback.kind is FeedbackKind.task_description
    assert traj.episodes[1].round == 1
    assert traj.episodes[1].feedback.kind is FeedbackKind.review_comment
    assert traj.episodes[1].prior_diff != ""
    assert traj.provenance.license == "MIT"
    assert traj.provenance.head_sha == "a" * 40


def test_mine_pr_trajectories_skips_excluded_license(fake_client) -> None:
    fake_client.get_repo_license.return_value = "AGPL-3.0"
    out = mine_pr_trajectories(
        "owner/repo",
        pr_numbers=[1],
        github_client=fake_client,
    )
    assert out == []
