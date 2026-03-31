"""GitHub trajectory mining for coding session distillation.

Mines PR diff chains and issue-commit chains from GitHub repositories,
producing trajectory dicts suitable for normalization and distillation.
Designed to run on an L4 VM with network access and a GITHUB_TOKEN.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from model_training.github_client import GitHubClient

logger = logging.getLogger(__name__)

__all__ = ["mine_pr_diff_chains", "mine_issue_commit_chains", "search_quality_prs"]

_FIXES_RE = re.compile(
    r"(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE
)

_DEFAULT_EXCLUDE_LABELS = frozenset(
    {
        "dependencies",
        "documentation",
        "docs",
        "chore",
        "ci",
        "bot",
    }
)


def search_quality_prs(
    repo: str,
    max_results: int = 100,
    github_token: str | None = None,
    min_review_comments: int = 1,
    min_commits: int = 2,
    exclude_labels: list[str] | None = None,
) -> list[int]:
    """Search for high-quality merged PRs using the GitHub Search API.

    Pre-filters PRs by review approval, comment count, label exclusion,
    and minimum commit count to identify PRs with meaningful review
    trajectories suitable for distillation.

    Args:
        repo: GitHub repository in "owner/repo" format.
        max_results: Maximum number of qualifying PR numbers to return.
        github_token: Personal access token for GitHub API authentication.
        min_review_comments: Minimum number of comments for search query.
        min_commits: Minimum number of commits a PR must have.
        exclude_labels: Labels to exclude. Defaults to common non-code labels.

    Returns:
        List of qualifying PR numbers.
    """
    client = GitHubClient(token=github_token)
    labels_to_exclude = (
        frozenset(exclude_labels)
        if exclude_labels is not None
        else _DEFAULT_EXCLUDE_LABELS
    )

    query = (
        f"repo:{repo} is:pr is:merged review:approved comments:>{min_review_comments}"
    )
    per_page = min(max_results, 100)
    pages_needed = math.ceil(max_results / 100)

    all_items: list[dict[str, Any]] = []
    for page in range(1, pages_needed + 1):
        data = client.get(
            "/search/issues",
            params={
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
        )
        items = data.get("items", [])
        all_items.extend(items)
        if len(items) < per_page:
            break

    total = len(all_items)

    # Label filter
    after_label: list[dict[str, Any]] = []
    for item in all_items:
        item_labels = {lbl["name"] for lbl in item.get("labels", [])}
        if not item_labels & labels_to_exclude:
            after_label.append(item)

    # Commit count filter
    result: list[int] = []
    for item in after_label:
        pr_number = item["number"]
        detail = client.get(f"/repos/{repo}/pulls/{pr_number}")
        if detail.get("commits", 0) >= min_commits:
            result.append(pr_number)
        if len(result) >= max_results:
            break

    logger.info(
        "Search found %d candidates, %d after label filter, %d after commit filter",
        total,
        len(after_label),
        len(result),
    )
    return result


def mine_pr_diff_chains(
    repo: str,
    max_prs: int = 100,
    github_token: str | None = None,
    pr_numbers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Extract PR diff chains from a GitHub repository.

    Each chain represents an iterative coding session: initial commit ->
    review comments -> revision commits. The resulting trajectory records
    capture the back-and-forth of code review as a multi-step improvement
    process, suitable for distillation.

    Returns trajectory dicts with the following fields:
    - task_id: f"pr_{repo}_{pr_number}"
    - task_description: PR title concatenated with body text
    - steps: list of commit diffs and review comments in chronological order
    - outcome: "merged" or "closed" depending on PR final state

    Args:
        repo: GitHub repository in "owner/repo" format.
        max_prs: Maximum number of PRs to process. Defaults to 100.
        github_token: Personal access token for GitHub API authentication.
        pr_numbers: Optional list of specific PR numbers to mine. When
            provided, skips the paginated PR list fetch and fetches each
            PR individually.

    Returns:
        List of trajectory dicts representing PR diff chains.
    """
    client = GitHubClient(token=github_token)

    if pr_numbers is not None:
        prs = [client.get(f"/repos/{repo}/pulls/{n}") for n in pr_numbers[:max_prs]]
    else:
        prs = client.get_paginated(
            f"/repos/{repo}/pulls",
            params={"state": "closed", "sort": "updated", "direction": "desc"},
            max_pages=math.ceil(max_prs / 100),
        )
        prs = prs[:max_prs]

    if not prs:
        return []

    trajectories: list[dict[str, Any]] = []

    for pr in prs:
        pr_number = pr["number"]
        title = pr.get("title", "")
        body = pr.get("body", "") or ""

        commits = client.get_paginated(
            f"/repos/{repo}/pulls/{pr_number}/commits",
            max_pages=5,
        )
        reviews = client.get_paginated(
            f"/repos/{repo}/pulls/{pr_number}/comments",
            max_pages=3,
        )

        steps: list[dict[str, str]] = []

        for commit in commits:
            sha = commit["sha"]
            msg = commit["commit"]["message"]
            detail = client.get(f"/repos/{repo}/commits/{sha}")
            files = detail.get("files", [])
            patches = []
            for f in files:
                patch = f.get("patch", "")
                if patch:
                    patches.append(f"--- {f['filename']} ---\n{patch}")
            steps.append(
                {
                    "type": "commit",
                    "description": msg,
                    "content": "\n".join(patches),
                }
            )

        for comment in reviews:
            steps.append(
                {
                    "type": "review",
                    "description": "Review comment",
                    "content": comment.get("body", ""),
                }
            )

        outcome = "merged" if pr.get("merged_at") is not None else "closed"

        trajectories.append(
            {
                "task_id": f"pr_{repo}_{pr_number}",
                "task_description": f"{title}\n\n{body}".strip(),
                "steps": steps,
                "outcome": outcome,
            }
        )

    return trajectories


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
    - task_id: f"issue_{repo}_{issue_number}"
    - task_description: issue title concatenated with body text
    - steps: list of commits referencing this issue in chronological order
    - outcome: "closed" or "open" from the issue state

    Args:
        repo: GitHub repository in "owner/repo" format.
        max_issues: Maximum number of issues to process. Defaults to 100.
        github_token: Personal access token for GitHub API authentication.

    Returns:
        List of trajectory dicts representing issue-commit chains.
    """
    client = GitHubClient(token=github_token)

    raw_issues = client.get_paginated(
        f"/repos/{repo}/issues",
        params={"state": "all", "sort": "updated", "direction": "desc"},
        max_pages=math.ceil(max_issues / 100),
    )

    # Filter out pull requests (GitHub issues API includes PRs)
    issues = [i for i in raw_issues if not i.get("pull_request")]
    issues = issues[:max_issues]

    issue_numbers = {i["number"] for i in issues}
    issue_map = {i["number"]: i for i in issues}

    repo_commits = client.get_paginated(
        f"/repos/{repo}/commits",
        max_pages=10,
    )

    # Group commits by referenced issue number
    linked: dict[int, list[dict[str, Any]]] = {}
    for commit in repo_commits:
        msg = commit["commit"]["message"]
        refs = _FIXES_RE.findall(msg)
        for ref in refs:
            issue_num = int(ref)
            if issue_num in issue_numbers:
                linked.setdefault(issue_num, []).append(commit)

    trajectories: list[dict[str, Any]] = []

    for issue_num, commits in linked.items():
        issue = issue_map[issue_num]
        title = issue.get("title", "")
        body = issue.get("body", "") or ""

        steps: list[dict[str, str]] = []
        for commit in commits:
            sha = commit["sha"]
            msg = commit["commit"]["message"]
            detail = client.get(f"/repos/{repo}/commits/{sha}")
            files = detail.get("files", [])
            patches = []
            for f in files:
                patch = f.get("patch", "")
                if patch:
                    patches.append(f"--- {f['filename']} ---\n{patch}")
            steps.append(
                {
                    "type": "commit",
                    "description": msg,
                    "content": "\n".join(patches),
                }
            )

        trajectories.append(
            {
                "task_id": f"issue_{repo}_{issue_num}",
                "task_description": f"{title}\n\n{body}".strip(),
                "steps": steps,
                "outcome": issue.get("state", "open"),
            }
        )

    return trajectories
