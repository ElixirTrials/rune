"""GitHub trajectory mining for coding session distillation.

Mines PR diff chains and issue-commit chains from GitHub repositories,
producing trajectory dicts suitable for normalization and distillation.
Designed to run on an L4 VM with network access and a GITHUB_TOKEN.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from model_training.d2l_feedback import (
    extract_failure_summary,
    truncate_head_tail,
)
from model_training.d2l_licenses import LicenseStatus, classify_license
from model_training.d2l_models import (
    Anchor,
    Episode,
    Feedback,
    FeedbackKind,
    Provenance,
    Trajectory,
)
from model_training.d2l_pairing import FeedbackEvent, pair_feedback_with_commits
from model_training.github_client import GitHubClient

logger = logging.getLogger(__name__)

__all__ = [
    "mine_pr_diff_chains",
    "mine_issue_commit_chains",
    "mine_pr_trajectories",
    "search_quality_prs",
    "search_quality_prs_v2",
    "score_pr_quality",
]

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

        # Build timestamped steps for chronological interleaving
        timed_steps: list[tuple[str, dict[str, str]]] = []

        for commit in commits:
            sha = commit["sha"]
            msg = commit["commit"]["message"]
            ts = commit["commit"].get("committer", {}).get("date", "")
            detail = client.get(f"/repos/{repo}/commits/{sha}")
            files = detail.get("files", [])
            patches = []
            for f in files:
                patch = f.get("patch", "")
                if patch:
                    patches.append(f"--- {f['filename']} ---\n{patch}")
            timed_steps.append(
                (
                    ts,
                    {
                        "type": "commit",
                        "description": msg,
                        "content": "\n".join(patches),
                    },
                )
            )

        for comment in reviews:
            ts = comment.get("created_at", "")
            timed_steps.append(
                (
                    ts,
                    {
                        "type": "review",
                        "description": "Review comment",
                        "content": comment.get("body", ""),
                    },
                )
            )

        # Sort by timestamp so commits and reviews interleave chronologically
        timed_steps.sort(key=lambda x: x[0])
        steps = [step for _, step in timed_steps]

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


_BOT_LOGINS = frozenset(
    {"dependabot[bot]", "renovate[bot]", "github-actions[bot]"}
)
_NONCODE_LABELS = frozenset(
    {"documentation", "docs", "chore", "ci", "dependencies"}
)


def score_pr_quality(pr_features: dict[str, Any]) -> float:
    """Compute a corrective-richness score for one PR.

    Hard exclusions return 0.0:
    - merged_at is None (unmerged)
    - author login is a known bot
    - any label in the no-code denylist

    Otherwise the score is a weighted sum:
    - +2.0 per anchored review comment (capped at 5)
    - +1.5 per CI failure that was resolved by a subsequent commit
    - +1.0 if 3 ≤ n_commits ≤ 12 (sweet-spot bonus)
    - -0.05 * max(0, p95_files_per_commit - 20) (mass-edit penalty)
    """
    if not pr_features.get("merged_at"):
        return 0.0
    user = pr_features.get("user") or {}
    if user.get("login") in _BOT_LOGINS:
        return 0.0
    label_names = {lbl["name"] for lbl in pr_features.get("labels", [])}
    if label_names & _NONCODE_LABELS:
        return 0.0

    anchored = min(int(pr_features.get("review_comments_with_anchor", 0)), 5)
    ci_resolved = int(pr_features.get("ci_failures_resolved", 0))
    n_commits = int(pr_features.get("n_commits", 0))
    p95_files = int(pr_features.get("n_files_changed_per_commit_p95", 0))

    score = 2.0 * anchored + 1.5 * ci_resolved
    if 3 <= n_commits <= 12:
        score += 1.0
    score -= 0.05 * max(0, p95_files - 20)
    return max(0.0, score)


_PATCH_LINE_CAP = 2000


def _aggregate_patch(files: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for f in files:
        patch = f.get("patch", "")
        if not patch:
            continue
        lines = patch.splitlines()
        if len(lines) > _PATCH_LINE_CAP:
            continue
        parts.append(f"--- {f['filename']} ---\n{patch}")
    return "\n".join(parts)


def _failed_check_to_event(
    run: dict[str, Any], ts: datetime
) -> FeedbackEvent | None:
    body = (run.get("output") or {}).get("summary") or run.get("name", "")
    body = truncate_head_tail(body, max_bytes=4096)
    name = (run.get("name") or "").lower()
    hint = "lint" if "lint" in name or "ruff" in name or "eslint" in name else None
    if hint is None and "build" in name:
        hint = "build"
    summary, anchor, kind = extract_failure_summary(body, hint=hint)
    return FeedbackEvent(
        kind=kind.value,
        body=body,
        ts=ts,
        author=None,
        anchor=anchor,
    )


def mine_pr_trajectories(
    repo: str,
    *,
    pr_numbers: list[int] | None = None,
    max_prs: int = 100,
    github_token: str | None = None,
    github_client: Any | None = None,
) -> list[Trajectory]:
    """Mine ``repo`` into a list of :class:`Trajectory` records."""
    client = github_client or GitHubClient(token=github_token)

    spdx = client.get_repo_license(repo)
    if classify_license(spdx) is LicenseStatus.excluded:
        logger.info("Skipping %s: license %r excluded by whitelist", repo, spdx)
        return []

    if pr_numbers is not None:
        prs = [client.get(f"/repos/{repo}/pulls/{n}") for n in pr_numbers[:max_prs]]
    else:
        prs = client.get_paginated(
            f"/repos/{repo}/pulls",
            params={"state": "closed", "sort": "updated", "direction": "desc"},
            max_pages=math.ceil(max_prs / 100),
        )[:max_prs]

    out: list[Trajectory] = []
    for pr in prs:
        if pr.get("merged_at") is None:
            continue
        traj = _build_trajectory(client, repo, pr, spdx or "")
        if traj is not None:
            out.append(traj)
    return out


def _features_for_pr(client: Any, repo: str, pr_number: int) -> dict[str, Any]:
    """Fetch the score-input features for one PR."""
    detail = client.get(f"/repos/{repo}/pulls/{pr_number}")
    review_comments = client.get_paginated(
        f"/repos/{repo}/pulls/{pr_number}/comments", max_pages=3
    )
    n_anchored = sum(1 for c in review_comments if c.get("path"))

    commits = client.get_paginated(
        f"/repos/{repo}/pulls/{pr_number}/commits", max_pages=5
    )
    p95_files = 0
    if commits:
        files_per_commit = []
        for c in commits:
            d = client.get(f"/repos/{repo}/commits/{c['sha']}")
            files_per_commit.append(len(d.get("files", [])))
        files_per_commit.sort()
        idx = max(0, int(0.95 * len(files_per_commit)) - 1)
        p95_files = files_per_commit[idx]

    n_ci_failed = 0
    for c in commits:
        n_ci_failed += len(client.get_check_runs(repo, c["sha"], only_failed=True))

    return {
        "user": detail.get("user") or {},
        "review_comments_with_anchor": n_anchored,
        "n_commits": len(commits),
        "ci_failures_resolved": n_ci_failed,
        "labels": detail.get("labels") or [],
        "n_files_changed_per_commit_p95": p95_files,
        "merged_at": detail.get("merged_at"),
    }


def search_quality_prs_v2(
    repo: str,
    max_results: int = 100,
    github_token: str | None = None,
) -> list[int]:
    """Return PR numbers ranked by corrective richness, top-``max_results``."""
    client = GitHubClient(token=github_token)
    items_data = client.get(
        "/search/issues",
        params={
            "q": f"repo:{repo} is:pr is:merged review:approved",
            "sort": "updated",
            "order": "desc",
            "per_page": min(max_results * 2, 100),
        },
    )
    items = items_data.get("items", [])
    label_names_per_pr = {
        item["number"]: {lbl["name"] for lbl in item.get("labels", [])}
        for item in items
    }

    candidates: list[tuple[float, int]] = []
    for item in items:
        pr_number = item["number"]
        if label_names_per_pr[pr_number] & _NONCODE_LABELS:
            continue
        feats = _features_for_pr(client, repo, pr_number)
        score = score_pr_quality(feats)
        if score > 0:
            candidates.append((score, pr_number))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out = [pr for _score, pr in candidates[:max_results]]
    logger.info("search_quality_prs_v2(%s) → %d PRs", repo, len(out))
    return out


def _build_trajectory(
    client: Any,
    repo: str,
    pr: dict[str, Any],
    spdx: str,
) -> Trajectory | None:
    pr_number = pr["number"]
    pr_author = (pr.get("user") or {}).get("login") or ""

    commits = client.get_paginated(
        f"/repos/{repo}/pulls/{pr_number}/commits", max_pages=5
    )
    if not commits:
        return None

    hydrated: list[dict[str, Any]] = []
    for c in commits:
        detail = client.get(f"/repos/{repo}/commits/{c['sha']}")
        c = dict(c)
        c["files"] = detail.get("files", [])
        hydrated.append(c)
    commits = hydrated

    review_comments = client.get_paginated(
        f"/repos/{repo}/pulls/{pr_number}/comments", max_pages=3
    )
    fb_events: list[FeedbackEvent] = []
    for c in review_comments:
        body = truncate_head_tail(c.get("body") or "", max_bytes=4096)
        fb_events.append(
            FeedbackEvent(
                kind=FeedbackKind.review_comment.value,
                body=body,
                ts=datetime.fromisoformat(
                    c["created_at"].replace("Z", "+00:00")
                ),
                author=(c.get("user") or {}).get("login"),
                anchor=Anchor(file=c.get("path"), line=c.get("line")),
            )
        )

    for c in commits:
        ts = datetime.fromisoformat(
            c["commit"]["committer"]["date"].replace("Z", "+00:00")
        )
        for run in client.get_check_runs(repo, c["sha"], only_failed=True):
            ev = _failed_check_to_event(run, ts=ts)
            if ev is not None:
                fb_events.append(ev)

    rounds = pair_feedback_with_commits(commits, fb_events, pr_author=pr_author)

    title = pr.get("title", "")
    body = pr.get("body") or ""

    initial_diff = _aggregate_patch(commits[0].get("files", []))
    episodes: list[Episode] = [
        Episode(
            round=0,
            prior_diff="",
            feedback=Feedback(
                kind=FeedbackKind.task_description,
                body=truncate_head_tail(f"{title}\n\n{body}".strip(), 4096),
            ),
            action_diff=initial_diff,
        )
    ]
    cumulative = initial_diff
    for round_link in rounds:
        kind = FeedbackKind(round_link.feedback.kind)
        action_diff = _aggregate_patch(round_link.next_commit.get("files", []))
        episodes.append(
            Episode(
                round=len(episodes),
                prior_diff=cumulative,
                feedback=Feedback(
                    kind=kind,
                    body=round_link.feedback.body,
                    summary=(
                        extract_failure_summary(
                            round_link.feedback.body,
                            hint="lint" if kind is FeedbackKind.lint else None,
                        )[0]
                        if kind is not FeedbackKind.review_comment
                        else None
                    ),
                    author=round_link.feedback.author,
                    anchor=round_link.feedback.anchor,
                ),
                action_diff=action_diff,
            )
        )
        cumulative = cumulative + "\n" + action_diff if cumulative else action_diff

    if len(episodes) < 2:
        return None

    return Trajectory(
        task_id=f"pr_{repo}_{pr_number}",
        task_description=truncate_head_tail(f"{title}\n\n{body}".strip(), 4096),
        episodes=episodes,
        metadata={
            "outcome": "merged",
            "n_rounds": len(episodes),
            "n_commits": len(commits),
            "labels": [lbl["name"] for lbl in pr.get("labels", [])],
        },
        provenance=Provenance(
            repo=repo,
            pr_number=pr_number,
            license=spdx,
            head_sha=pr["head"]["sha"],
            base_sha=pr["base"]["sha"],
            mined_at=datetime.now(timezone.utc),
        ),
    )
