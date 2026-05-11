"""Thin GitHub REST + GraphQL API client with auth, pagination, and rate-limit retry.

Designed for batch data mining on a training VM (sync httpx is fine).
"""

from __future__ import annotations

import logging
import math
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
        """GET with retry on rate-limit (403) and server errors (5xx).

        Args:
            url: Full URL to request.
            params: Optional query parameters.
            max_retries: Maximum number of retries on transient errors.

        Returns:
            The successful httpx.Response.

        Raises:
            httpx.HTTPStatusError: On non-retryable error responses.
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
            if resp.status_code >= 500 and attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "Server error %d, retrying in %ds (attempt %d/%d)",
                    resp.status_code,
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

    def get_repo_license(self, repo: str) -> str | None:
        """Return the SPDX identifier for ``repo`` (or None when unlicensed).

        The result is cached per-instance so a batch mine over many PRs in
        the same repo only pays for one HTTP request.
        """
        cache: dict[str, str | None] = self.__dict__.setdefault("_license_cache", {})
        if repo in cache:
            return cache[repo]
        data = self.get(f"/repos/{repo}")
        spdx: str | None = None
        license_block = data.get("license")
        if isinstance(license_block, dict):
            spdx = license_block.get("spdx_id")
            if spdx == "NOASSERTION":
                spdx = None
        cache[repo] = spdx
        return spdx

    def get_check_runs(
        self,
        repo: str,
        sha: str,
        only_failed: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch check-runs for one commit. Optionally filter to failures."""
        data = self.get(f"/repos/{repo}/commits/{sha}/check-runs")
        runs: list[dict[str, Any]] = list(data.get("check_runs", []))
        if only_failed:
            runs = [r for r in runs if r.get("conclusion") == "failure"]
        return runs

    def get_check_suites(self, repo: str, sha: str) -> list[dict[str, Any]]:
        """Fetch check-suites for one commit (used for build_failure detection)."""
        data = self.get(f"/repos/{repo}/commits/{sha}/check-suites")
        return list(data.get("check_suites", []))

    def get_issue_comments(self, repo: str, issue_number: int) -> list[dict[str, Any]]:
        """Top-level issue/PR comments (the "conversation" tab, not inline reviews)."""
        return self.get_paginated(
            f"/repos/{repo}/issues/{issue_number}/comments",
            max_pages=3,
        )

    # ------------------------------------------------------------------ #
    # GraphQL API                                                          #
    # ------------------------------------------------------------------ #

    _GQL_URL = "https://api.github.com/graphql"

    def graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """POST a GraphQL query and return the ``data`` dict.

        Args:
            query: GraphQL query string.
            variables: Optional variable bindings.
            max_retries: Retries on rate-limit or server errors.

        Returns:
            The ``data`` dict from the GraphQL response.

        Raises:
            RuntimeError: If the response contains GraphQL errors.
            httpx.HTTPStatusError: On HTTP-level errors.
        """
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(max_retries + 1):
            resp = httpx.post(
                self._GQL_URL,
                headers=self._headers,
                json=payload,
                timeout=30.0,
            )
            if resp.status_code >= 500 and attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "GQL server error %d, retrying in %ds (attempt %d/%d)",
                    resp.status_code,
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            body = resp.json()
            errors = body.get("errors", [])
            if errors:
                msg = errors[0].get("message", "")
                if "rate limit" in msg.lower() and attempt < max_retries:
                    wait = int(resp.headers.get("Retry-After", "60"))
                    logger.warning(
                        "GQL rate limited, sleeping %ds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"GraphQL errors: {errors}")
            return body.get("data") or {}
        # unreachable in normal flow
        raise RuntimeError("graphql: exhausted retries")  # pragma: no cover

    # -- search + score PRs in one GraphQL round-trip ------------------- #

    _SEARCH_PRS_GQL = """
query($query: String!, $first: Int!, $after: String) {
  search(query: $query, type: ISSUE, first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        mergedAt
        author { login }
        labels(first: 10) { nodes { name } }
        commits(first: 20) {
          totalCount
          nodes {
            commit {
              changedFilesIfAvailable
              statusCheckRollup {
                contexts(first: 10) {
                  nodes {
                    ... on CheckRun {
                      conclusion
                    }
                  }
                }
              }
            }
          }
        }
        reviews(first: 10) {
          totalCount
          nodes {
            comments(first: 10) {
              nodes { path }
            }
          }
        }
      }
    }
  }
}
"""

    @staticmethod
    def _gql_node_to_features(node: dict[str, Any]) -> dict[str, Any]:
        """Convert one GraphQL search node to a scoring feature dict."""
        commits_data = node.get("commits", {})
        commit_nodes = commits_data.get("nodes", [])
        n_commits = commits_data.get("totalCount", len(commit_nodes))

        files_per_commit: list[int] = []
        n_ci_failed = 0
        for cn in commit_nodes:
            commit = cn.get("commit", {})
            n_files = commit.get("changedFilesIfAvailable") or 0
            files_per_commit.append(n_files)
            rollup = commit.get("statusCheckRollup") or {}
            for ctx in (rollup.get("contexts") or {}).get("nodes", []):
                if ctx.get("conclusion") == "failure":
                    n_ci_failed += 1

        p95_files = 0
        if files_per_commit:
            files_per_commit.sort()
            idx = max(0, int(0.95 * len(files_per_commit)) - 1)
            p95_files = files_per_commit[idx]

        n_anchored = 0
        for rev in (node.get("reviews") or {}).get("nodes", []):
            for c in (rev.get("comments") or {}).get("nodes", []):
                if c.get("path"):
                    n_anchored += 1

        author_login = (node.get("author") or {}).get("login") or ""
        labels = [
            {"name": lbl["name"]} for lbl in (node.get("labels") or {}).get("nodes", [])
        ]

        return {
            "number": node.get("number"),
            "merged_at": node.get("mergedAt"),
            "user": {"login": author_login},
            "labels": labels,
            "review_comments_with_anchor": n_anchored,
            "n_commits": n_commits,
            "ci_failures_resolved": n_ci_failed,
            "n_files_changed_per_commit_p95": p95_files,
        }

    def search_and_score_prs_graphql(
        self,
        repo: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch merged PRs with scoring features via a single GraphQL query.

        Returns a list of feature dicts compatible with :func:`score_pr_quality`.
        Replaces ``search_quality_prs_v2`` + ``_features_for_pr`` — one GraphQL
        paginated query instead of ~6 REST calls per PR candidate.

        Args:
            repo: ``owner/name`` repository slug.
            max_results: Upper bound on candidates to fetch (rounded up to the
                next multiple of 100 for pagination alignment).

        Returns:
            List of dicts with keys matching what ``score_pr_quality`` expects.
        """
        search_query = f"repo:{repo} is:pr is:merged review:approved"
        per_page = 25
        pages_needed = math.ceil(max_results / per_page)
        cursor: str | None = None
        results: list[dict[str, Any]] = []

        for _ in range(pages_needed):
            variables: dict[str, Any] = {
                "query": search_query,
                "first": per_page,
            }
            if cursor is not None:
                variables["after"] = cursor

            data = self.graphql(self._SEARCH_PRS_GQL, variables)
            search = data.get("search", {})
            nodes = search.get("nodes", [])

            for node in nodes:
                if not node:
                    continue
                results.append(self._gql_node_to_features(node))

            page_info = search.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

        return results

    # -- fetch full PR metadata (all except file patches) --------------- #

    _PR_METADATA_GQL = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      title
      body
      mergedAt
      headRefOid
      baseRefOid
      author { login }
      labels(first: 10) { nodes { name } }
      reviews(first: 100) {
        nodes {
          comments(first: 50) {
            nodes {
              body
              path
              line
              author { login }
              createdAt
            }
          }
        }
      }
      commits(first: 100) {
        nodes {
          commit {
            oid
            committedDate
            statusCheckRollup {
              contexts(first: 50) {
                nodes {
                  ... on CheckRun {
                    name
                    conclusion
                    text
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

    def fetch_pr_metadata_graphql(
        self,
        repo: str,
        pr_number: int,
    ) -> dict[str, Any]:
        """Fetch all PR metadata except file patches in one GraphQL call.

        File patches still require REST (GraphQL doesn't expose them).

        Args:
            repo: ``owner/name`` repository slug.
            pr_number: Pull-request number.

        Returns:
            Parsed PR metadata dict with keys: ``title``, ``body``,
            ``merged_at``, ``head_sha``, ``base_sha``, ``author``,
            ``labels``, ``review_comments``, ``commits``.
        """
        owner, name = repo.split("/", 1)
        data = self.graphql(
            self._PR_METADATA_GQL,
            {"owner": owner, "name": name, "number": pr_number},
        )
        pr = (data.get("repository") or {}).get("pullRequest") or {}

        # Normalise commits
        commits: list[dict[str, Any]] = []
        for cn in (pr.get("commits") or {}).get("nodes", []):
            commit = cn.get("commit") or {}
            check_runs: list[dict[str, Any]] = []
            rollup = commit.get("statusCheckRollup") or {}
            for ctx in (rollup.get("contexts") or {}).get("nodes", []):
                # Only CheckRun nodes have conclusion
                if "conclusion" in ctx:
                    check_runs.append(
                        {
                            "name": ctx.get("name", ""),
                            "conclusion": ctx.get("conclusion"),
                            # GraphQL exposes text instead of output.summary
                            "output": {"summary": ctx.get("text") or ""},
                        }
                    )
            commits.append(
                {
                    "oid": commit.get("oid", ""),
                    "committed_date": commit.get("committedDate", ""),
                    "check_runs": check_runs,
                }
            )

        # Normalise review comments (nested: reviews → comments)
        review_comments: list[dict[str, Any]] = []
        for rev in (pr.get("reviews") or {}).get("nodes", []):
            for c in (rev.get("comments") or {}).get("nodes", []):
                review_comments.append(
                    {
                        "body": c.get("body", ""),
                        "path": c.get("path"),
                        "line": c.get("line"),
                        "user": {"login": (c.get("author") or {}).get("login")},
                        "created_at": c.get("createdAt", ""),
                    }
                )

        labels = [
            {"name": lbl["name"]} for lbl in (pr.get("labels") or {}).get("nodes", [])
        ]
        author_login = (pr.get("author") or {}).get("login") or ""

        return {
            "title": pr.get("title", ""),
            "body": pr.get("body") or "",
            "merged_at": pr.get("mergedAt"),
            "head_sha": pr.get("headRefOid", ""),
            "base_sha": pr.get("baseRefOid", ""),
            "author": {"login": author_login},
            "labels": labels,
            "review_comments": review_comments,
            "commits": commits,
        }
