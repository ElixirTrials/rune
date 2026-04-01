"""Thin GitHub REST API client with auth, pagination, and rate-limit retry.

Designed for batch data mining on a training VM (sync httpx is fine).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

__all__ = ["GitHubClient"]

_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


class GitHubClient:
    """Minimal GitHub REST API client.

    Handles authentication, paginated list endpoints, and automatic
    retry on rate-limit 403 responses.

    Args:
        token: GitHub personal access token. Optional for public
            endpoints but required for private repos and higher rate
            limits.
        base_url: API base URL. Override for GitHub Enterprise.
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
    ) -> None:
        """Initialize the client with optional auth token.

        Args:
            token: GitHub personal access token.
            base_url: API base URL.
        """
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token is not None:
            self._headers["Authorization"] = f"Bearer {token}"

    def _get_response(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> httpx.Response:
        """GET with rate-limit retry, returning the raw response.

        Args:
            url: Full URL to request.
            params: Optional query parameters.
            max_retries: Maximum number of retries on rate-limit 403.

        Returns:
            The successful httpx.Response.

        Raises:
            httpx.HTTPStatusError: On non-rate-limit error responses.
        """
        for attempt in range(max_retries + 1):
            resp = httpx.get(
                url,
                headers=self._headers,
                params=params,
                timeout=30.0,
            )
            if resp.status_code == 403:
                try:
                    body = resp.json()
                except (ValueError, KeyError):
                    logger.debug("Non-JSON 403 body, skip rate-limit check")
                    body = {}
                message = body.get("message", "")
                if "rate limit" in message.lower() and attempt < max_retries:
                    wait = int(resp.headers.get("Retry-After", "60"))
                    logger.warning(
                        "Rate limited, sleeping %ds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            return resp
        # Unreachable in normal flow, but satisfies type checker.
        resp.raise_for_status()  # pragma: no cover
        return resp  # pragma: no cover

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> Any:
        """GET a single API endpoint with rate-limit retry.

        Args:
            path: API path relative to base_url (e.g. ``/repos/owner/repo``).
            params: Optional query parameters.
            max_retries: Maximum number of retries on rate-limit 403.

        Returns:
            Parsed JSON response body.

        Raises:
            httpx.HTTPStatusError: On non-rate-limit error responses.
        """
        url = f"{self._base_url}{path}"
        return self._get_response(url, params=params, max_retries=max_retries).json()

    def get_paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        max_pages: int = 10,
        per_page: int = 100,
    ) -> list[Any]:
        """GET a paginated list endpoint, following Link rel=next headers.

        Args:
            path: API path relative to base_url.
            params: Optional query parameters (``per_page`` is injected).
            max_pages: Maximum number of pages to fetch.
            per_page: Items per page (max 100 for most GitHub endpoints).

        Returns:
            Flat list of all items across all fetched pages.
        """
        merged_params: dict[str, Any] = dict(params or {})
        merged_params["per_page"] = per_page

        items: list[Any] = []
        url: str | None = f"{self._base_url}{path}"

        for _ in range(max_pages):
            if url is None:
                break
            resp = self._get_response(url, params=merged_params)
            items.extend(resp.json())

            # After the first request, params are baked into the Link URL.
            merged_params = {}

            link = resp.headers.get("Link", "")
            match = _LINK_NEXT_RE.search(link)
            url = match.group(1) if match else None

        return items
