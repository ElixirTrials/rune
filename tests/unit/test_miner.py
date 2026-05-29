"""Unit tests for the mining pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from rune.mining.miner import (
    extract_trajectories,
    load_session,
    mine_corpus,
    scan_sessions,
)


def _make_session(
    base: Path,
    name: str,
    steps: list[dict],  # type: ignore[type-arg]
    metadata: dict,  # type: ignore[type-arg]
) -> Path:
    session_dir = base / name
    session_dir.mkdir(parents=True)
    (session_dir / "session.jsonl").write_text("\n".join(json.dumps(s) for s in steps))
    (session_dir / "metadata.json").write_text(json.dumps(metadata))
    return session_dir


_STEPS = [
    {
        "step": 0,
        "action": "decompose",
        "target": None,
        "input": "implement binary search",
        "output": "subtasks: a, b",
        "feedback": None,
    },
    {
        "step": 1,
        "action": "plan",
        "target": "subtask_a",
        "input": "plan subtask_a",
        "output": "plan text",
        "feedback": None,
    },
    {
        "step": 2,
        "action": "code",
        "target": "subtask_a",
        "input": "code subtask_a",
        "output": "def bs(): ...",
        "feedback": {"exit_code": 0, "stdout": "ok", "stderr": ""},
    },
]

_META = {
    "task": "implement binary search",
    "benchmark": "humaneval",
    "problem_id": "HE_001",
    "timestamp": "2026-01-01T00:00:00",
}


def test_scan_sessions(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    _make_session(sessions_dir, "s1", _STEPS, _META)
    _make_session(sessions_dir, "s2", _STEPS, _META)
    found = scan_sessions(sessions_dir)
    assert len(found) == 2
    assert all(p.name in ("s1", "s2") for p in found)


def test_scan_sessions_empty(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    assert scan_sessions(sessions_dir) == []


def test_load_session(tmp_path: Path) -> None:
    session_dir = _make_session(tmp_path, "s1", _STEPS, _META)
    steps, metadata = load_session(session_dir)
    assert len(steps) == 3
    assert steps[0]["action"] == "decompose"
    assert metadata["benchmark"] == "humaneval"


def test_load_session_no_metadata(tmp_path: Path) -> None:
    session_dir = tmp_path / "s1"
    session_dir.mkdir()
    (session_dir / "session.jsonl").write_text(json.dumps(_STEPS[0]))
    steps, metadata = load_session(session_dir)
    assert len(steps) == 1
    assert metadata == {}


def test_extract_trajectories(tmp_path: Path) -> None:
    records = extract_trajectories(_STEPS, _META)
    actions = [r["metadata"]["phase"] for r in records]
    assert set(actions) == {"decompose", "plan", "code"}
    for r in records:
        assert r["task_id"] == "humaneval/HE_001"
        assert r["metadata"]["benchmark"] == "humaneval"
        assert r["metadata"]["problem_id"] == "HE_001"
        assert "trajectory" in r
        assert isinstance(r["trajectory"], str)


def test_extract_trajectories_trajectory_text() -> None:
    records = extract_trajectories(_STEPS, _META)
    code_record = next(r for r in records if r["metadata"]["phase"] == "code")
    assert "def bs():" in code_record["trajectory"]
    assert "exit_code" in code_record["trajectory"]


def test_mine_corpus(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    output_dir = tmp_path / "corpus"
    _make_session(sessions_dir, "s1", _STEPS, _META)

    steps2 = [
        {
            "step": 0,
            "action": "code",
            "target": None,
            "input": "mbpp input",
            "output": "mbpp output",
            "feedback": None,
        }
    ]
    meta2 = {
        "task": "sort list",
        "benchmark": "mbpp",
        "problem_id": "MBPP_42",
        "timestamp": "2026-01-02T00:00:00",
    }
    _make_session(sessions_dir, "s2", steps2, meta2)

    counts = mine_corpus(sessions_dir, output_dir)

    assert "code_humaneval" in counts
    assert "code_mbpp" in counts
    assert counts["code_humaneval"] == 1
    assert counts["code_mbpp"] == 1
    assert counts["decompose_humaneval"] == 1
    assert counts["plan_humaneval"] == 1

    shard = output_dir / "code_humaneval.jsonl"
    assert shard.exists()
    lines = [json.loads(ln) for ln in shard.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["task_id"] == "humaneval/HE_001"


def test_extract_trajectories_no_subtask_conflation() -> None:
    # Two subtasks each have a `code` step. They must yield two separate code
    # records keyed on (action, target), not one conflated record.
    steps = [
        {"step": 0, "action": "code", "target": "subtask_a",
         "input": "code a", "output": "def a(): pass", "feedback": None},
        {"step": 1, "action": "code", "target": "subtask_b",
         "input": "code b", "output": "def b(): pass", "feedback": None},
    ]
    records = [r for r in extract_trajectories(steps, _META)
               if r["metadata"]["phase"] == "code"]
    assert len(records) == 2
    targets = {r["metadata"]["target"] for r in records}
    assert targets == {"subtask_a", "subtask_b"}
    rec_a = next(r for r in records if r["metadata"]["target"] == "subtask_a")
    assert "def a():" in rec_a["trajectory"]
    assert "def b():" not in rec_a["trajectory"]  # not conflated


def test_mine_corpus_creates_output_dir(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    output_dir = tmp_path / "does" / "not" / "exist"
    counts = mine_corpus(sessions_dir, output_dir)
    assert output_dir.exists()
    assert counts == {}
