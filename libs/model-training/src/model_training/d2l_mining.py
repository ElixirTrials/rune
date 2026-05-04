"""GitHub trajectory mining for coding session distillation.

Mines PRs into structured trajectory records suitable for hypernetwork
training. Each PR becomes one :class:`Trajectory` JSONL record.
"""

from __future__ import annotations

import logging
import math
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
    "mine_pr_trajectories",
    "search_quality_prs_v2",
    "score_pr_quality",
]

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
