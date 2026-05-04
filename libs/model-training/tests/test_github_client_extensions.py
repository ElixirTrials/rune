"""Tests for new GitHubClient endpoints (license, check-runs, etc)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from model_training.github_client import GitHubClient


def _mock_response(json_body: dict | list, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_body
    resp.headers = headers or {}
    resp.raise_for_status.return_value = None
    return resp


def test_get_repo_license_returns_spdx() -> None:
    client = GitHubClient(token="x")
    payload = {"license": {"spdx_id": "Apache-2.0", "name": "Apache License 2.0"}}
    with patch("httpx.get", return_value=_mock_response(payload)):
        assert client.get_repo_license("owner/repo") == "Apache-2.0"


def test_get_repo_license_caches_per_repo() -> None:
    client = GitHubClient(token="x")
    payload = {"license": {"spdx_id": "MIT"}}
    with patch("httpx.get", return_value=_mock_response(payload)) as mocked:
        client.get_repo_license("owner/repo")
        client.get_repo_license("owner/repo")
        assert mocked.call_count == 1


def test_get_repo_license_returns_none_when_unlicensed() -> None:
    client = GitHubClient(token="x")
    with patch("httpx.get", return_value=_mock_response({"license": None})):
        assert client.get_repo_license("owner/repo") is None


def test_get_check_runs_returns_failed_runs_only() -> None:
    client = GitHubClient(token="x")
    payload = {
        "check_runs": [
            {"name": "ci", "conclusion": "success", "output": {"summary": "ok"}},
            {"name": "tests", "conclusion": "failure", "output": {"summary": "boom"}},
        ]
    }
    with patch("httpx.get", return_value=_mock_response(payload)):
        runs = client.get_check_runs("owner/repo", "a" * 40, only_failed=True)
        assert len(runs) == 1
        assert runs[0]["name"] == "tests"
