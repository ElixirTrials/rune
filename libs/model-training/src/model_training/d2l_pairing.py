"""Pair feedback events to the next commit they motivated.

Pairing rule:
- ``review_comment``: the next commit by the PR author after the comment ts.
- ``ci_failure`` / ``test_failure`` / ``lint`` / ``build_failure``: the next
  commit by anyone after the event ts (the failure can be fixed by a
  collaborator pushing to the branch).

A feedback event with no subsequent matching commit is dropped — that is a PR
where the reviewer's request was never addressed, which carries no corrective
signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from model_training.d2l_models import Anchor

__all__ = ["FeedbackEvent", "RoundLink", "pair_feedback_with_commits"]


@dataclass(frozen=True)
class FeedbackEvent:
    kind: str
    body: str
    ts: datetime
    author: str | None
    anchor: Anchor | None


@dataclass(frozen=True)
class RoundLink:
    feedback: FeedbackEvent
    next_commit: dict[str, Any]


_AUTHOR_KINDS = frozenset({"review_comment"})


def _commit_ts(commit: dict[str, Any]) -> datetime:
    iso = commit["commit"]["committer"]["date"]
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _commit_login(commit: dict[str, Any]) -> str | None:
    author = commit.get("author")
    if isinstance(author, dict):
        return author.get("login")
    return None


def pair_feedback_with_commits(
    commits: list[dict[str, Any]],
    feedback: list[FeedbackEvent],
    pr_author: str,
) -> list[RoundLink]:
    """Return one :class:`RoundLink` per feedback event that is followed by a
    matching commit.

    Commits are walked in order. Each feedback event consumes the next
    commit that satisfies the kind-specific pairing rule and is later than
    the event timestamp.
    """
    sorted_commits = sorted(commits, key=_commit_ts)
    sorted_fb = sorted(feedback, key=lambda f: f.ts)

    consumed = 0
    rounds: list[RoundLink] = []
    for fb in sorted_fb:
        i = consumed
        while i < len(sorted_commits):
            commit = sorted_commits[i]
            if _commit_ts(commit) <= fb.ts:
                i += 1
                continue
            if fb.kind in _AUTHOR_KINDS and _commit_login(commit) != pr_author:
                i += 1
                continue
            rounds.append(RoundLink(feedback=fb, next_commit=commit))
            consumed = i + 1
            break
        else:
            continue  # no matching commit; drop this feedback event
    return rounds
