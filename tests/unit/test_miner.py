"""Unit tests for the mining pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    {"step": 0, "action": "decompose", "target": None,
     "trajectory": "ROLE: decomposer", "prompt": "implement binary search",
     "output": "subtasks: a, b", "feedback": None},
    {"step": 1, "action": "plan", "target": "subtask_a",
     "trajectory": "ROLE: planner", "prompt": "plan subtask_a",
     "output": "plan text", "feedback": None},
    {"step": 2, "action": "code", "target": "subtask_a",
     "trajectory": "ROLE: coder", "prompt": "code subtask_a",
     "output": "def bs(): ...", "feedback": {"exit_code": 0, "stdout": "ok", "stderr": ""}},
]

_META = {
    "task": "implement binary search",
    "benchmark": "humaneval",
    "problem_id": "HE_001",
    "timestamp": "2026-01-01T00:00:00",
    "schema_version": 2,
}

_STEPS_FAIL = [
    {"step": 0, "action": "code", "target": "recovered_subtask",
     "trajectory": "t", "prompt": "p", "output": "bad code",
     "feedback": {"exit_code": 1, "stdout": "", "stderr": "err"}},
    {"step": 1, "action": "diagnose", "target": "recovered_subtask",
     "trajectory": "t", "prompt": "p", "output": "diagnosis", "feedback": None},
    {"step": 2, "action": "repair", "target": "recovered_subtask",
     "trajectory": "t", "prompt": "p", "output": "fixed code",
     "feedback": {"exit_code": 0, "stdout": "ok", "stderr": ""}},
    {"step": 3, "action": "code", "target": "failed_subtask",
     "trajectory": "t", "prompt": "p", "output": "bad",
     "feedback": {"exit_code": 1, "stdout": "", "stderr": "err"}},
    {"step": 4, "action": "diagnose", "target": "failed_subtask",
     "trajectory": "t", "prompt": "p", "output": "diagnosis2", "feedback": None},
    {"step": 5, "action": "repair", "target": "failed_subtask",
     "trajectory": "t", "prompt": "p", "output": "still bad",
     "feedback": {"exit_code": 1, "stdout": "", "stderr": "err"}},
]


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
        assert r["task_id"].startswith("humaneval/HE_001/")
        assert r["metadata"]["benchmark"] == "humaneval"
        assert r["metadata"]["problem_id"] == "HE_001"
        assert "trajectory" in r
        assert isinstance(r["trajectory"], str)


def test_extract_trajectories_trajectory_text() -> None:
    records = extract_trajectories(_STEPS, _META)
    code_record = next(r for r in records if r["metadata"]["phase"] == "code")
    assert code_record["trajectory"] == "ROLE: coder"
    assert "def bs():" in code_record["completion"]


def test_passed_run_keeps_one_record_per_step() -> None:
    meta = {"benchmark": "mbpp", "problem_id": "1", "pass_at_1": True, "schema_version": 2}
    recs = extract_trajectories(_STEPS, meta)
    assert len(recs) == len(_STEPS)                      # one per step, no join
    r0 = recs[0]
    assert r0["prompt"] and r0["completion"]             # coherent single pair
    assert "\n---\n" not in r0["completion"]             # never concatenated
    assert r0["metadata"]["pass_at_1"] is True


def test_failed_run_keeps_only_recovered_diagnose() -> None:
    meta = {"benchmark": "mbpp", "problem_id": "2", "pass_at_1": False, "schema_version": 2}
    recs = extract_trajectories(_STEPS_FAIL, meta)
    assert recs, "recovered-repair diagnose traces must survive a failed run"
    assert all(r["metadata"]["phase"] == "diagnose" for r in recs)
    assert all(r["metadata"]["target"] == "recovered_subtask" for r in recs)


def test_unknown_verdict_keeps_all() -> None:
    meta = {"benchmark": "smoke", "problem_id": "x", "schema_version": 2}  # no pass_at_1
    recs = extract_trajectories(_STEPS, meta)
    assert len(recs) == len(_STEPS)                      # smoke corpus: no scoring -> keep all


def test_wrong_schema_version_fails_fast() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        extract_trajectories(_STEPS, {"benchmark": "mbpp", "problem_id": "1", "schema_version": 1})


def test_mine_corpus(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    output_dir = tmp_path / "corpus"
    _make_session(sessions_dir, "s1", _STEPS, _META)

    steps2 = [
        {"step": 0, "action": "code", "target": None,
         "trajectory": "ROLE: coder", "prompt": "mbpp input",
         "output": "mbpp output", "feedback": None}
    ]
    meta2 = {
        "task": "sort list",
        "benchmark": "mbpp",
        "problem_id": "MBPP_42",
        "timestamp": "2026-01-02T00:00:00",
        "schema_version": 2,
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
    assert lines[0]["task_id"].startswith("humaneval/HE_001/")


def test_extract_trajectories_no_subtask_conflation() -> None:
    # Two subtasks each have a `code` step. They must yield two separate code
    # records, not one conflated record.
    steps = [
        {"step": 0, "action": "code", "target": "subtask_a",
         "trajectory": "ROLE: coder", "prompt": "code a",
         "output": "def a(): pass", "feedback": None},
        {"step": 1, "action": "code", "target": "subtask_b",
         "trajectory": "ROLE: coder", "prompt": "code b",
         "output": "def b(): pass", "feedback": None},
    ]
    records = [
        r
        for r in extract_trajectories(steps, _META)
        if r["metadata"]["phase"] == "code"
    ]
    assert len(records) == 2
    targets = {r["metadata"]["target"] for r in records}
    assert targets == {"subtask_a", "subtask_b"}
    rec_a = next(r for r in records if r["metadata"]["target"] == "subtask_a")
    assert "def a():" in rec_a["completion"]
    assert "def b():" not in rec_a["completion"]  # not conflated


def test_mine_corpus_creates_output_dir(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    output_dir = tmp_path / "does" / "not" / "exist"
    counts = mine_corpus(sessions_dir, output_dir)
    assert output_dir.exists()
    assert counts == {}
