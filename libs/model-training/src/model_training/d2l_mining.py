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
_ACTION_DIFF_CHAR_CAP = 30_000
_PRIOR_DIFF_WINDOW = 8_000


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


def search_quality_prs_v2(
    repo: str,
    max_results: int = 100,
    github_token: str | None = None,
) -> list[int]:
    """Return PR numbers ranked by corrective richness, top-``max_results``.

    Uses GraphQL to fetch all candidate PRs and their scoring features in
    ~1 paginated query instead of ~6 REST calls per PR candidate.
    """
    client = GitHubClient(token=github_token)
    candidates_data = client.search_and_score_prs_graphql(
        repo, max_results=max_results * 2
    )

    candidates: list[tuple[float, int]] = []
    for feats in candidates_data:
        pr_number = feats.get("number")
        if pr_number is None:
            continue
        score = score_pr_quality(feats)
        if score > 0:
            candidates.append((score, pr_number))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out = [pr for _score, pr in candidates[:max_results]]
    logger.info("search_quality_prs_v2(%s) -> %d PRs", repo, len(out))
    return out


def _hydrate_commits(
    client: Any, repo: str, gql_commits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fetch file patches for each commit via REST and merge with GQL data."""
    hydrated: list[dict[str, Any]] = []
    for c in gql_commits:
        sha = c["oid"]
        detail = client.get(f"/repos/{repo}/commits/{sha}")
        hydrated.append(
            {
                **c,
                "files": detail.get("files", []),
                "sha": sha,
                "author": detail.get("author"),
                "commit": {"committer": {"date": c.get("committed_date", "")}},
            }
        )
    return hydrated


def _feedback_events_from_gql(
    gql_meta: dict[str, Any],
    hydrated: list[dict[str, Any]],
) -> list[FeedbackEvent]:
    """Build FeedbackEvent list from GraphQL PR metadata (no extra REST calls)."""
    fb_events: list[FeedbackEvent] = []

    for c in gql_meta.get("review_comments", []):
        comment_author = (c.get("user") or {}).get("login")
        body = truncate_head_tail(c.get("body") or "", max_bytes=4096)
        created = c.get("created_at", "")
        fb_events.append(
            FeedbackEvent(
                kind=FeedbackKind.review_comment.value,
                body=body,
                ts=datetime.fromisoformat(created.replace("Z", "+00:00")),
                author=comment_author,
                anchor=Anchor(file=c.get("path"), line=c.get("line")),
            )
        )

    for c in hydrated:
        raw_date = c.get("committed_date") or ""
        ts = (
            datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            if raw_date
            else datetime.now(timezone.utc)
        )
        for run in c.get("check_runs", []):
            if run.get("conclusion") != "failure":
                continue
            ev = _failed_check_to_event(run, ts=ts)
            if ev is not None:
                fb_events.append(ev)

    return fb_events


def _build_trajectory(
    client: Any,
    repo: str,
    pr: dict[str, Any],
    spdx: str,
) -> Trajectory | None:
    """Build a :class:`Trajectory` for one PR.

    ``pr`` is either a raw REST PR dict (legacy path via ``mine_pr_trajectories``
    when ``pr_numbers`` are supplied) or a normalised GraphQL metadata dict.

    File patches are always fetched via REST — GraphQL doesn't expose them.
    """
    pr_number = pr["number"]
    pr_author = (
        (pr.get("user") or {}).get("login")
        or (pr.get("author") or {}).get("login")
        or ""
    )

    gql_meta = client.fetch_pr_metadata_graphql(repo, pr_number)
    gql_commits = gql_meta.get("commits", [])
    if not gql_commits:
        return None

    hydrated = _hydrate_commits(client, repo, gql_commits)
    fb_events = _feedback_events_from_gql(gql_meta, hydrated)
    rounds = pair_feedback_with_commits(hydrated, fb_events, pr_author=pr_author)

    title = pr.get("title") or gql_meta.get("title", "")
    body = pr.get("body") or gql_meta.get("body") or ""

    initial_diff = _aggregate_patch(hydrated[0].get("files", []))
    if len(initial_diff) > _ACTION_DIFF_CHAR_CAP:
        initial_diff = initial_diff[:_ACTION_DIFF_CHAR_CAP]
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
        if len(action_diff) > _ACTION_DIFF_CHAR_CAP:
            continue
        windowed_prior = cumulative[-_PRIOR_DIFF_WINDOW:] if cumulative else ""
        episodes.append(
            Episode(
                round=len(episodes),
                prior_diff=windowed_prior,
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

    # Prefer head/base from GQL metadata; fall back to REST PR shape.
    head_sha = (
        gql_meta.get("head_sha")
        or (pr.get("head") or {}).get("sha", "")
    )
    base_sha = (
        gql_meta.get("base_sha")
        or (pr.get("base") or {}).get("sha", "")
    )

    labels = pr.get("labels") or gql_meta.get("labels", [])

    return Trajectory(
        task_id=f"pr_{repo}_{pr_number}",
        task_description=truncate_head_tail(f"{title}\n\n{body}".strip(), 4096),
        episodes=episodes,
        metadata={
            "outcome": "merged",
            "n_rounds": len(episodes),
            "n_commits": len(hydrated),
            "labels": [lbl["name"] for lbl in labels],
        },
        provenance=Provenance(
            repo=repo,
            pr_number=pr_number,
            license=spdx,
            head_sha=head_sha,
            base_sha=base_sha,
            mined_at=datetime.now(timezone.utc),
        ),
    )
