"""Trajectory recording and formatting for coding session distillation.

Provides functions to persist, load, and convert coding session trajectories
into SFT-compatible chat format for LoRA fine-tuning pipelines.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SYSTEM_PROMPT = "You are a Python code generator. Output only code, no explanation."


def _get_trajectory_dir() -> Path:
    """Return the trajectory storage directory, respecting RUNE_TRAJECTORY_DIR env var.

    Reads env var inside function body (not module level) so that monkeypatch.setenv()
    works correctly in tests.
    """
    env_dir = os.environ.get("RUNE_TRAJECTORY_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".rune" / "trajectories"


def record_trajectory(
    session_id: str,
    steps: list[dict[str, Any]],
    outcome: Optional[str] = None,
    *,
    task_description: str = "",
    task_type: str = "",
    adapter_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Persist a coding session trajectory to disk for future distillation.

    Args:
        session_id: Unique identifier for the coding session.
        steps: List of step dicts, each containing attempt results.
        outcome: Final session result ('success', 'exhausted', or None).
        task_description: Natural language description of the coding task.
        task_type: Category of task (e.g. 'function', 'class', 'refactor').
        adapter_ids: LoRA adapter IDs used during the session.

    Returns:
        A dict with 'session_id' and 'file_path' keys.
    """
    trajectory_dir = _get_trajectory_dir()
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    file_path = trajectory_dir / f"{session_id}.json"
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    trajectory: dict[str, Any] = {
        "session_id": session_id,
        "task_description": task_description,
        "task_type": task_type,
        "adapter_ids": adapter_ids if adapter_ids is not None else [],
        "outcome": outcome,
        "timestamp": timestamp,
        "steps": steps,
    }

    file_path.write_text(json.dumps(trajectory, indent=2))

    return {"session_id": session_id, "file_path": str(file_path)}


def load_trajectory(trajectory_id: str) -> dict[str, Any]:
    """Load a stored trajectory by session ID.

    Args:
        trajectory_id: The session ID used as the filename (without .json).

    Returns:
        A dict containing the full trajectory data including steps and metadata.

    Raises:
        FileNotFoundError: If no trajectory file exists for the given ID.
    """
    trajectory_dir = _get_trajectory_dir()
    file_path = trajectory_dir / f"{trajectory_id}.json"
    # Let FileNotFoundError propagate naturally if file does not exist
    return json.loads(file_path.read_text())  # type: ignore[no-any-return]


def format_for_sft(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a trajectory into SFT-compatible chat format.

    Only successful trajectories (outcome == 'success') produce output.
    Extracts the final step where tests_passed is True as the assistant message.

    Args:
        trajectory: A trajectory dict as returned by load_trajectory.

    Returns:
        A list of 3 message dicts ([system, user, assistant]) for successful
        trajectories, or an empty list if the trajectory did not succeed.
    """
    if trajectory.get("outcome") != "success":
        return []

    steps: list[dict[str, Any]] = trajectory.get("steps", [])
    successful_step = next(
        (s for s in reversed(steps) if s.get("tests_passed")),
        None,
    )

    if successful_step is None:
        return []

    task_description: str = trajectory.get("task_description", "")
    generated_code: str = successful_step.get("generated_code", "")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_description},
        {"role": "assistant", "content": generated_code},
    ]
