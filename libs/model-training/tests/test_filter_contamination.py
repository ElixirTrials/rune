"""Tests for the post-hoc contamination filter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import filter_contamination as fc  # noqa: E402
from build_benchmark_fingerprints import fingerprint  # noqa: E402


def _write_traj(p: Path, repo: str, body: str) -> None:
    rec = {
        "task_id": f"pr_{repo}_1",
        "task_description": body,
        "episodes": [],
        "metadata": {},
        "provenance": {
            "repo": repo, "pr_number": 1, "license": "MIT",
            "head_sha": "a"*40, "base_sha": "b"*40,
            "mined_at": "2026-05-01T00:00:00+00:00",
        },
    }
    with p.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def test_drops_traj_whose_description_matches_fingerprint(tmp_path) -> None:
    fp_file = tmp_path / "fp.json"
    fp_file.write_text(json.dumps({
        "humaneval_plus": [fingerprint("solve sudoku")],
        "swebench_lite_repos": [],
    }))
    in_file = tmp_path / "in.jsonl"
    out_file = tmp_path / "out.jsonl"
    _write_traj(in_file, "owner/repo", "solve  sudoku")
    _write_traj(in_file, "owner/repo", "build a website")
    fc.filter_corpus(in_file, out_file, fp_file)
    kept = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(kept) == 1
    assert kept[0]["task_description"] == "build a website"


def test_drops_traj_from_swebench_repo(tmp_path) -> None:
    fp_file = tmp_path / "fp.json"
    fp_file.write_text(json.dumps({
        "humaneval_plus": [],
        "swebench_lite_repos": ["sympy/sympy"],
    }))
    in_file = tmp_path / "in.jsonl"
    out_file = tmp_path / "out.jsonl"
    _write_traj(in_file, "sympy/sympy", "anything")
    _write_traj(in_file, "owner/repo", "anything")
    fc.filter_corpus(in_file, out_file, fp_file)
    kept = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(kept) == 1
    assert kept[0]["provenance"]["repo"] == "owner/repo"
