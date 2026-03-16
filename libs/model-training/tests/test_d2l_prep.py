"""Tests for the d2l_prep data preparation pipeline.

Covers prepare_training_jsonl (filters failed trajectories, writes JSONL)
and the CLI __main__ entry point.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SUCCESSFUL_TRAJECTORY: dict = {
    "task_id": "task-001",
    "task_description": "Write a function that returns 42.",
    "steps": [
        {
            "description": "Wrote the function body.",
            "tests_passed": True,
            "canonical_solution": "def answer():\n    return 42",
        }
    ],
    "outcome": "success",
}

FAILED_TRAJECTORY: dict = {
    "task_id": "task-002",
    "task_description": "Write a function that returns the string hello.",
    "steps": [
        {
            "description": "Attempted solution.",
            "tests_passed": False,
            "generated_code": "def answer():\n    return 'goodbye'",
        }
    ],
    "outcome": "failure",
}


# ---------------------------------------------------------------------------
# Test 1: successful trajectory produces records, failed trajectory is filtered
# ---------------------------------------------------------------------------


def test_prepare_training_jsonl_filters_failures(tmp_path: Path) -> None:
    """Records from failed trajectories are excluded; successful ones are included."""
    from model_training.d2l_prep import prepare_training_jsonl

    success_file = tmp_path / "success.json"
    failure_file = tmp_path / "failure.json"
    output_file = tmp_path / "output.jsonl"

    success_file.write_text(json.dumps(SUCCESSFUL_TRAJECTORY))
    failure_file.write_text(json.dumps(FAILED_TRAJECTORY))

    count = prepare_training_jsonl(
        input_paths=[success_file, failure_file],
        output_path=output_file,
    )

    assert output_file.exists(), "output file must be created"
    lines = [line for line in output_file.read_text().splitlines() if line.strip()]
    assert count == len(lines), "returned count must match line count"
    assert count > 0, "successful trajectory must produce at least one record"

    for line in lines:
        record = json.loads(line)
        assert "activation_text" in record
        assert "teacher_text" in record
        assert "task_id" in record


# ---------------------------------------------------------------------------
# Test 2: no successful trajectories → empty file, count 0
# ---------------------------------------------------------------------------


def test_prepare_training_jsonl_all_failures_empty_output(tmp_path: Path) -> None:
    """When all trajectories fail, output JSONL has zero records."""
    from model_training.d2l_prep import prepare_training_jsonl

    failure_file = tmp_path / "failure.json"
    output_file = tmp_path / "output.jsonl"

    failure_file.write_text(json.dumps(FAILED_TRAJECTORY))

    count = prepare_training_jsonl(
        input_paths=[failure_file],
        output_path=output_file,
    )

    assert count == 0, "zero records expected for all-failure input"
    assert output_file.exists(), "output file must be created even when empty"
    lines = [line for line in output_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 0


# ---------------------------------------------------------------------------
# Test 3: CLI --help exits 0
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    """Running python -m model_training.d2l_prep --help exits with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "model_training.d2l_prep", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower(), (
        "help output must contain 'usage'"
    )
