"""Smoke test for the public surface of d2l_mining after the trajectory rewrite."""

from __future__ import annotations

from model_training.d2l_mining import (
    mine_pr_trajectories,
    score_pr_quality,
    search_quality_prs_v2,
)


def test_public_api_exists() -> None:
    assert callable(mine_pr_trajectories)
    assert callable(search_quality_prs_v2)
    assert callable(score_pr_quality)
