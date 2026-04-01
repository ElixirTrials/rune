"""Tests for batch mining mode in mine_github.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


def _make_config(tmp_path: Path, repos: list[dict[str, Any]] | None = None) -> Path:
    """Write a minimal repos config and return its path."""
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


def _make_trajectory(repo: str, number: int) -> dict[str, Any]:
    """Create a minimal mined trajectory."""
    return {
        "task_id": f"pr_{repo}_{number}",
        "task_description": f"PR #{number}",
        "steps": [
            {"type": "commit", "description": "Init", "content": f"+code_{number}"},
            {"type": "review", "description": "R", "content": "Fix it"},
            {"type": "commit", "description": "Fix", "content": f"+fixed_{number}"},
        ],
        "outcome": "merged",
    }


@patch("model_training.d2l_mining.mine_pr_diff_chains")
def test_run_batch_produces_per_repo_jsonl(
    mock_mine: MagicMock, tmp_path: Path
) -> None:
    """Batch mode produces one JSONL file per repo."""
    config_path = _make_config(tmp_path)
    output_dir = tmp_path / "output"

    mock_mine.side_effect = lambda repo, **kw: [_make_trajectory(repo, 1)]

    # Import the batch function from the script
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
        compress=True,
        max_diff_lines=500,
    )

    # Check output files
    assert (output_dir / "test_alpha.jsonl").exists()
    assert (output_dir / "test_beta.jsonl").exists()

    # Verify JSONL content
    from model_training.d2l_data import load_jsonl

    alpha_records = load_jsonl(output_dir / "test_alpha.jsonl")
    assert len(alpha_records) == 2  # step_0 + step_1 from one trajectory
    assert alpha_records[0]["task_id"] == "pr_test/alpha_1"
    assert "metadata" in alpha_records[0]


def test_mining_repos_config_is_valid_json() -> None:
    """The committed repos config parses without error."""
    config_path = (
        Path(__file__).resolve().parents[3] / "instructions" / "mining_repos.json"
    )
    if not config_path.exists():
        return  # skip if not yet created
    config = json.loads(config_path.read_text())
    assert "repos" in config
    assert len(config["repos"]) > 0
    for entry in config["repos"]:
        assert "repo" in entry
        assert "/" in entry["repo"]
