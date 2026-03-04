"""Trajectory recording and formatting stubs for coding session distillation.

All functions raise NotImplementedError. No GPU imports required.
"""

from __future__ import annotations

from typing import Any, Optional


def record_trajectory(
    session_id: str,
    steps: list[dict[str, Any]],
    outcome: Optional[str] = None,
) -> dict[str, Any]:
    """Persist a coding session trajectory for future distillation.

    Args:
        session_id: Unique identifier for the coding session.
        steps: List of step dicts, each containing action, observation, and reflection.
        outcome: Final session result ('success', 'exhausted', or None).

    Returns:
        A dict containing the trajectory ID and storage metadata.

    Raises:
        NotImplementedError: Method is not yet implemented.

    Example:
        >>> steps = [{"action": "edit", "observation": "err"}]
        >>> result = record_trajectory("session-001", steps, outcome="success")
        >>> result["trajectory_id"]  # Returns trajectory ID when implemented
        'traj-001'
    """
    raise NotImplementedError(
        "record_trajectory is not yet implemented. "
        "It will persist a coding session trajectory for distillation."
    )


def load_trajectory(trajectory_id: str) -> dict[str, Any]:
    """Load a stored trajectory by its unique identifier.

    Args:
        trajectory_id: The unique ID of the trajectory to load.

    Returns:
        A dict containing the full trajectory data including steps and metadata.

    Raises:
        NotImplementedError: Method is not yet implemented.

    Example:
        >>> trajectory = load_trajectory("traj-001")
        >>> trajectory["session_id"]  # Returns session ID when implemented
        'session-001'
    """
    raise NotImplementedError(
        "load_trajectory is not yet implemented. "
        "It will load a stored trajectory by ID."
    )


def format_for_sft(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a trajectory into SFT-compatible chat format.

    Transforms a raw coding trajectory into the conversation format
    expected by supervised fine-tuning pipelines.

    Args:
        trajectory: A trajectory dict as returned by load_trajectory.

    Returns:
        A list of message dicts with 'role' and 'content' keys.

    Raises:
        NotImplementedError: Method is not yet implemented.

    Example:
        >>> messages = format_for_sft(trajectory)
        >>> all("role" in m for m in messages)
        True
    """
    raise NotImplementedError(
        "format_for_sft is not yet implemented. "
        "It will convert a trajectory into SFT-compatible chat format."
    )
