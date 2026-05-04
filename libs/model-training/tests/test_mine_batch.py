"""Tests for batch mining mode in mine_github.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from model_training.d2l_models import (
    Episode,
    Feedback,
    FeedbackKind,
    Provenance,
    Trajectory,
)


def _make_config(tmp_path: Path, repos: list[dict[str, Any]] | None = None) -> Path:
    config = {
        "defaults": {"max_prs": 5, "quality": False},
        "repos": repos
        or [
            {"repo": "test/alpha", "language": "python"},
            {"repo": "test/beta", "language": "go"},
        ],
    }
    config_path = tmp_path / "repos.json"
    config_path.write_text(json.dumps(config))
    return config_path


def _make_trajectory(repo: str, number: int) -> Trajectory:
    from datetime import datetime, timezone

    return Trajectory(
        task_id=f"pr_{repo}_{number}",
        task_description=f"PR #{number}",
        episodes=[
            Episode(
                round=0,
                prior_diff="",
                feedback=Feedback(kind=FeedbackKind.task_description, body="goal"),
                action_diff="d0",
            ),
            Episode(
                round=1,
                prior_diff="d0",
                feedback=Feedback(
                    kind=FeedbackKind.review_comment, body="fix it", author="rev"
                ),
                action_diff="d1",
            ),
        ],
        metadata={"outcome": "merged"},
        provenance=Provenance(
            repo=repo,
            pr_number=number,
            license="MIT",
            head_sha="a" * 40,
            base_sha="b" * 40,
            mined_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
    )


@patch("model_training.d2l_mining.mine_pr_trajectories")
def test_run_batch_produces_per_repo_jsonl(
    mock_mine: MagicMock, tmp_path: Path
) -> None:
    config_path = _make_config(tmp_path)
    output_dir = tmp_path / "output"

    mock_mine.side_effect = lambda repo, **kw: [_make_trajectory(repo, 1)]

    import importlib.util
    import sys

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "mine_github.py"
    spec = importlib.util.spec_from_file_location("mine_github", script_path)
    assert spec and spec.loader
    mine_github = importlib.util.module_from_spec(spec)
    sys.modules["mine_github"] = mine_github
    spec.loader.exec_module(mine_github)

    mine_github._run_batch(
        config_path=config_path,
        output_dir=output_dir,
        token="fake-token",
    )

    assert (output_dir / "test_alpha.trajectories.jsonl").exists()
    assert (output_dir / "test_beta.trajectories.jsonl").exists()
    assert (output_dir / "test_alpha.unrolled.jsonl").exists()

    from model_training.d2l_data import load_jsonl

    pairs = load_jsonl(output_dir / "test_alpha.unrolled.jsonl")
    assert len(pairs) == 2
    assert pairs[0]["task_id"] == "pr_test/alpha_1"


def test_mining_repos_config_is_valid_json() -> None:
    config_path = (
        Path(__file__).resolve().parents[3] / "instructions" / "mining_repos.json"
    )
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    assert "repos" in config
    assert len(config["repos"]) > 0
    for entry in config["repos"]:
        assert "repo" in entry
        assert "/" in entry["repo"]
