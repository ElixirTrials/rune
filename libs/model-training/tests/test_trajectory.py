"""Green-phase behavior tests for model_training.trajectory module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from model_training.trajectory import format_for_sft, load_trajectory, record_trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_STEPS: list[dict[str, Any]] = [
    {
        "attempt": 0,
        "generated_code": "def add(a, b):\n    return a + b",
        "stdout": "",
        "stderr": "AssertionError",
        "exit_code": 1,
        "tests_passed": False,
    },
    {
        "attempt": 1,
        "generated_code": "def add(a, b):\n    return a + b\n\nassert add(1, 2) == 3",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": True,
    },
]


# ---------------------------------------------------------------------------
# record_trajectory tests
# ---------------------------------------------------------------------------


def test_record_trajectory_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory writes a JSON file to the configured trajectory directory."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    record_trajectory("sess-001", SAMPLE_STEPS, outcome="success")
    expected_file = tmp_path / "sess-001.json"
    assert expected_file.exists(), "Expected JSON file was not created"
    data = json.loads(expected_file.read_text())
    assert data["session_id"] == "sess-001"
    assert data["outcome"] == "success"
    assert "timestamp" in data
    assert len(data["steps"]) == 2


def test_record_trajectory_returns_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory returns dict with session_id and file_path keys."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    result = record_trajectory("sess-002", [], outcome="exhausted")
    assert result["session_id"] == "sess-002"
    assert "file_path" in result
    assert Path(result["file_path"]).exists()


def test_record_trajectory_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory creates the directory if it does not exist."""
    subdir = tmp_path / "nested" / "trajectories"
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(subdir))
    assert not subdir.exists(), "Directory should not exist yet"
    record_trajectory("sess-003", [], outcome="success")
    assert subdir.exists(), "Directory should have been created"
    assert (subdir / "sess-003.json").exists()


def test_record_trajectory_includes_task_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_trajectory includes task_description, task_type, and adapter_ids."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    record_trajectory(
        "sess-004",
        SAMPLE_STEPS,
        outcome="success",
        task_description="Write an add function",
        task_type="function",
        adapter_ids=["adapter-001", "adapter-002"],
    )
    data = json.loads((tmp_path / "sess-004.json").read_text())
    assert data["task_description"] == "Write an add function"
    assert data["task_type"] == "function"
    assert data["adapter_ids"] == ["adapter-001", "adapter-002"]


# ---------------------------------------------------------------------------
# load_trajectory tests
# ---------------------------------------------------------------------------


def test_load_trajectory_reads_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_trajectory reads back a previously recorded trajectory (round-trip)."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    record_trajectory(
        "sess-005",
        SAMPLE_STEPS,
        outcome="success",
        task_description="Round-trip test",
        task_type="function",
        adapter_ids=["adapter-abc"],
    )
    loaded = load_trajectory("sess-005")
    assert loaded["session_id"] == "sess-005"
    assert loaded["outcome"] == "success"
    assert loaded["task_description"] == "Round-trip test"
    assert len(loaded["steps"]) == 2


def test_load_trajectory_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_trajectory raises FileNotFoundError for non-existent session_id."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_trajectory("does-not-exist")


# ---------------------------------------------------------------------------
# format_for_sft tests
# ---------------------------------------------------------------------------


def _make_trajectory(
    outcome: str,
    steps: list[dict[str, Any]],
    task_description: str = "Write an add function",
) -> dict[str, Any]:
    """Helper to create a trajectory dict for format_for_sft tests."""
    return {
        "session_id": "sess-test",
        "outcome": outcome,
        "task_description": task_description,
        "steps": steps,
    }


def test_format_for_sft_success() -> None:
    """format_for_sft returns [system, user, assistant] for a successful trajectory."""
    trajectory = _make_trajectory(
        "success", SAMPLE_STEPS, task_description="Write an add function"
    )
    messages = format_for_sft(trajectory)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Write an add function"
    assert messages[2]["role"] == "assistant"
    # assistant content should be from the last step with tests_passed=True
    assert "tests_passed" not in messages[2]["content"]  # it's code, not a dict
    assert "def add" in messages[2]["content"]


def test_format_for_sft_exhausted_returns_empty() -> None:
    """format_for_sft returns empty list for trajectories with outcome != 'success'."""
    trajectory = _make_trajectory("exhausted", SAMPLE_STEPS)
    result = format_for_sft(trajectory)
    assert result == []


def test_format_for_sft_no_successful_step_returns_empty() -> None:
    """format_for_sft returns [] when no step has tests_passed=True."""
    failing_steps: list[dict[str, Any]] = [
        {"attempt": 0, "generated_code": "broken", "tests_passed": False},
        {"attempt": 1, "generated_code": "still broken", "tests_passed": False},
    ]
    trajectory = _make_trajectory("success", failing_steps)
    result = format_for_sft(trajectory)
    assert result == []
