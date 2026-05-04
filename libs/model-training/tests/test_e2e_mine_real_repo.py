"""End-to-end mining against a small live repo. Skipped without GITHUB_TOKEN."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GITHUB_TOKEN"),
    reason="needs a live GITHUB_TOKEN",
)


def test_mine_three_prs_from_small_repo(tmp_path) -> None:
    from model_training.d2l_data import unroll_trajectory_to_pairs
    from model_training.d2l_mining import (
        mine_pr_trajectories,
        search_quality_prs_v2,
    )

    repo = "encode/httpx"
    pr_numbers = search_quality_prs_v2(
        repo, max_results=3, github_token=os.environ["GITHUB_TOKEN"]
    )
    if not pr_numbers:
        pytest.skip(f"no quality PRs found in {repo}")

    trajectories = mine_pr_trajectories(
        repo,
        pr_numbers=pr_numbers,
        github_token=os.environ["GITHUB_TOKEN"],
    )
    assert trajectories
    for traj in trajectories:
        assert traj.episodes[0].round == 0
        assert traj.provenance.license
        for ep in traj.episodes[1:]:
            assert ep.prior_diff != ""
        pairs = unroll_trajectory_to_pairs(traj)
        assert len(pairs) == len(traj.episodes)
