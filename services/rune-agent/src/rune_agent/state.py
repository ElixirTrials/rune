"""State definition for the Rune coding agent recursive loop."""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class RuneState(TypedDict):
    """State for the Rune coding agent recursive loop.

    Attributes:
        task_description: Natural language description of the coding task.
        task_type: Category of task (e.g. 'function', 'class', 'refactor').
        test_suite: Test code that the generated solution must pass.
        adapter_ids: LoRA adapter IDs to load for parametric memory.
        session_id: Unique identifier for trajectory persistence (UUID4).
        attempt_count: Current attempt number (0-indexed, incremented by reflect).
        max_attempts: Maximum number of generation attempts allowed.
        generated_code: Code produced by the generate node.
        stdout: Standard output from executing generated code.
        stderr: Standard error from executing generated code.
        exit_code: Process exit code from execution (0 = success).
        tests_passed: Whether the test suite passed on this attempt.
        trajectory: List of per-attempt records for parametric memory training.
        outcome: Terminal result -- 'success', 'exhausted', or None if still running.
    """

    # Task intake
    task_description: str
    task_type: str
    test_suite: str
    adapter_ids: list[str]
    # Loop tracking
    session_id: str
    attempt_count: int
    max_attempts: int
    # Per-attempt results
    generated_code: str
    stdout: str
    stderr: str
    exit_code: int
    tests_passed: bool
    # Accumulated trajectory
    trajectory: list[dict[str, Any]]
    # Pipeline phase
    phase: Optional[str]  # 'decompose' | 'plan' | 'code' | 'integrate' | None
    # Terminal result
    outcome: Optional[str]  # 'success' | 'exhausted' | None
