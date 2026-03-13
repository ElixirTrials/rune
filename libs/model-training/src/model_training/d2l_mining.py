"""GitHub trajectory mining for coding session distillation.

Designed to run on an L4 VM with network access, not in CI.
Stubs only — implementation runs on the training VM.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["mine_pr_diff_chains", "mine_issue_commit_chains"]


def mine_pr_diff_chains(
    repo: str,
    max_prs: int = 100,
    github_token: str | None = None,
) -> list[dict[str, Any]]:
    """Extract PR diff chains from a GitHub repository.

    Each chain represents an iterative coding session: initial commit ->
    review comments -> revision commits. The resulting trajectory records
    capture the back-and-forth of code review as a multi-step improvement
    process, suitable for distillation.

    Returns trajectory dicts with the following fields:
    - task_id: f"pr_{repo}_{pr_number}" (e.g., "pr_owner/repo_42")
    - task_description: PR title concatenated with body text
    - steps: list of commit diffs in chronological order
    - outcome: "merged" or "closed" depending on PR final state

    Args:
        repo: GitHub repository in "owner/repo" format (e.g., "python/cpython").
        max_prs: Maximum number of PRs to process. Defaults to 100.
        github_token: Personal access token for GitHub API authentication.
            Required for private repos and to avoid rate limiting.

    Returns:
        List of trajectory dicts representing PR diff chains.

    Raises:
        NotImplementedError: Always — run on L4 VM with GITHUB_TOKEN.
    """
    raise NotImplementedError("Run on L4 VM with GITHUB_TOKEN")


def mine_issue_commit_chains(
    repo: str,
    max_issues: int = 100,
    github_token: str | None = None,
) -> list[dict[str, Any]]:
    """Link GitHub issues to their fixing commits via commit message references.

    Scans commit messages for "fixes #N", "closes #N", or "resolves #N"
    patterns to identify which commits address which issues. Groups linked
    commits as trajectory steps for distillation.

    Returns trajectory dicts with the following fields:
    - task_id: f"issue_{repo}_{issue_number}" (e.g., "issue_owner/repo_123")
    - task_description: full issue body text
    - steps: list of commits referencing this issue in chronological order
    - outcome: "closed" when the issue was resolved, "open" if still open

    Args:
        repo: GitHub repository in "owner/repo" format (e.g., "python/cpython").
        max_issues: Maximum number of issues to process. Defaults to 100.
        github_token: Personal access token for GitHub API authentication.
            Required for private repos and to avoid rate limiting.

    Returns:
        List of trajectory dicts representing issue-commit chains.

    Raises:
        NotImplementedError: Always — run on L4 VM with GITHUB_TOKEN.
    """
    raise NotImplementedError("Run on L4 VM with GITHUB_TOKEN")
