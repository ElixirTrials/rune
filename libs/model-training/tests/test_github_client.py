"""Tests for model_training.github_client module."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
from model_training.github_client import GitHubClient


def _json_response(
    data: Any,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response with JSON body.

    Args:
        data: JSON-serializable payload.
        status_code: HTTP status code.
        headers: Optional response headers.
    """
    import json

    content = json.dumps(data).encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={
            "content-type": "application/json",
            **(headers or {}),
        },
        request=httpx.Request("GET", "https://api.github.com/test"),
    )


def test_auth_header_set_when_token_provided() -> None:
    """Authorization header is 'Bearer ghp_test123' when token given."""
    client = GitHubClient(token="ghp_test123")
    assert client._headers["Authorization"] == "Bearer ghp_test123"


def test_auth_header_absent_when_no_token() -> None:
    """No Authorization key in headers when no token provided."""
    client = GitHubClient()
    assert "Authorization" not in client._headers


@patch("model_training.github_client.httpx")
def test_get_returns_json(mock_httpx: Any) -> None:
    """get() returns parsed JSON from a successful response."""
    mock_httpx.get.return_value = _json_response({"id": 1, "name": "test"})
    client = GitHubClient(token="tok")
    result = client.get("/repos/owner/repo")

    assert result == {"id": 1, "name": "test"}
    mock_httpx.get.assert_called_once()


@patch("model_training.github_client.httpx")
def test_get_paginated_follows_link_header(mock_httpx: Any) -> None:
    """get_paginated follows Link rel=next across two pages."""
    page1 = _json_response(
        [{"id": 1}],
        headers={"Link": '<https://api.github.com/repos/o/r/pulls?page=2>; rel="next"'},
    )
    page2 = _json_response([{"id": 2}])
    mock_httpx.get.side_effect = [page1, page2]

    client = GitHubClient(token="tok")
    result = client.get_paginated("/repos/o/r/pulls")

    assert result == [{"id": 1}, {"id": 2}]
    assert mock_httpx.get.call_count == 2


@patch("model_training.github_client.httpx")
def test_get_paginated_respects_max_pages(mock_httpx: Any) -> None:
    """get_paginated stops after max_pages even if Link header present."""
    page1 = _json_response(
        [{"id": 1}],
        headers={"Link": '<https://api.github.com/repos/o/r/pulls?page=2>; rel="next"'},
    )
    mock_httpx.get.return_value = page1

    client = GitHubClient(token="tok")
    result = client.get_paginated("/repos/o/r/pulls", max_pages=1)

    assert result == [{"id": 1}]
    assert mock_httpx.get.call_count == 1


@patch("model_training.github_client.time")
@patch("model_training.github_client.httpx")
def test_rate_limit_retries_on_403(mock_httpx: Any, mock_time: Any) -> None:
    """get() retries on 403 rate-limit and returns result from second call."""
    rate_limited = _json_response(
        {"message": "API rate limit exceeded"},
        status_code=403,
        headers={"Retry-After": "0"},
    )
    success = _json_response({"ok": True})
    mock_httpx.get.side_effect = [rate_limited, success]

    client = GitHubClient(token="tok")
    result = client.get("/repos/owner/repo")

    assert result == {"ok": True}
    assert mock_httpx.get.call_count == 2
    mock_time.sleep.assert_called_once_with(0)
