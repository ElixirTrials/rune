"""Tests for corpus_stats — token / round-count distributions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import corpus_stats as cs  # noqa: E402


def _write_traj(p: Path, n_episodes: int, episode_len_chars: int) -> None:
    rec = {
        "task_id": f"pr_owner/repo_{n_episodes}",
        "task_description": "x" * episode_len_chars,
        "episodes": [
            {
                "round": i,
                "prior_diff": "" if i == 0 else "x" * episode_len_chars,
                "feedback": {"kind": "task_description", "body": "y" * 100},
                "action_diff": "z" * episode_len_chars,
            }
            for i in range(n_episodes)
        ],
        "metadata": {},
        "provenance": {
            "repo": "owner/repo",
            "pr_number": n_episodes,
            "license": "MIT",
            "head_sha": "a" * 40,
            "base_sha": "b" * 40,
            "mined_at": "2026-05-01T00:00:00+00:00",
        },
    }
    with p.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def test_stats_counts_trajectories_and_episodes(tmp_path) -> None:
    in_file = tmp_path / "in.jsonl"
    _write_traj(in_file, n_episodes=2, episode_len_chars=100)
    _write_traj(in_file, n_episodes=4, episode_len_chars=100)
    out_file = tmp_path / "stats.json"
    cs.compute_stats(in_file, out_file)
    stats = json.loads(out_file.read_text())
    assert stats["n_trajectories"] == 2
    assert stats["n_episodes"] == 6
    assert stats["rounds_per_traj"]["min"] == 2
    assert stats["rounds_per_traj"]["max"] == 4


def test_stats_p95_chars_close_to_max(tmp_path) -> None:
    in_file = tmp_path / "in.jsonl"
    for n in range(20):
        _write_traj(in_file, n_episodes=2, episode_len_chars=10 * (n + 1))
    out_file = tmp_path / "stats.json"
    cs.compute_stats(in_file, out_file)
    stats = json.loads(out_file.read_text())
    assert stats["chars_per_traj"]["p95"] >= stats["chars_per_traj"]["median"]
