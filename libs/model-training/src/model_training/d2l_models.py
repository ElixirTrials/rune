"""Pydantic schema for mined coding trajectories.

A `Trajectory` is one PR's corrective episodes — `(prior_diff, feedback,
action_diff)` triples in chronological order. Each PR becomes one JSONL
record at mine time; downstream training unrolls the episodes into
per-step SFT pairs via :func:`model_training.d2l_data.unroll_trajectory_to_pairs`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "Anchor",
    "Episode",
    "Feedback",
    "FeedbackKind",
    "Provenance",
    "Trajectory",
]


class FeedbackKind(str, Enum):
    """Kind of feedback signal in a corrective episode."""

    task_description = "task_description"
    review_comment = "review_comment"
    ci_failure = "ci_failure"
    test_failure = "test_failure"
    lint = "lint"
    build_failure = "build_failure"


class Anchor(BaseModel):
    """Where a feedback signal points in the codebase or test suite."""

    file: str | None = None
    line: int | None = None
    test: str | None = None


class Feedback(BaseModel):
    """One feedback event attached to one corrective round.

    `body` is the raw text (head+tail-truncated upstream).  `summary` is
    a heuristic-extracted one-line reflection — populated for ci/test/lint/
    build failures where the raw output is verbose. Review comments are
    already reflective and leave `summary` unset.
    """

    kind: FeedbackKind
    body: str
    summary: str | None = None
    author: str | None = None
    anchor: Anchor | None = None


class Episode(BaseModel):
    """One corrective round in a trajectory."""

    round: int = Field(ge=0)
    prior_diff: str
    feedback: Feedback
    action_diff: str


class Provenance(BaseModel):
    """Per-trajectory provenance for license-filter compliance (paper §B.4)."""

    repo: str
    pr_number: int
    license: str
    head_sha: str
    base_sha: str
    mined_at: datetime
    opt_out_revoked: bool = False

    @field_validator("head_sha", "base_sha")
    @classmethod
    def _sha_full_length(cls, v: str) -> str:
        if len(v) != 40 or not all(c in "0123456789abcdef" for c in v):
            raise ValueError(f"expected 40-char hex SHA, got {v!r}")
        return v


class Trajectory(BaseModel):
    """One PR's corrective episodes as a single JSONL record."""

    task_id: str
    task_description: str
    episodes: list[Episode]
    metadata: dict[str, Any]
    provenance: Provenance
