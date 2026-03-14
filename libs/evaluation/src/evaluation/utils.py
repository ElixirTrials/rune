"""Shared subprocess execution utilities for evaluation benchmarks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def safe_subprocess_run(
    script_path: Path,
    cwd: str,
    timeout: int = 30,
) -> bool:
    """Run a Python script in a subprocess and return whether it passed.

    Handles ``subprocess.TimeoutExpired`` by returning ``False``.
    Used by both HumanEval and OOD benchmark runners to avoid duplicated
    try/except + subprocess.run boilerplate.

    Args:
        script_path: Path to the Python script to execute.
        cwd: Working directory for the subprocess.
        timeout: Maximum execution time in seconds before the process is killed.

    Returns:
        ``True`` if the process exits with return code 0, ``False`` otherwise
        (including timeout).

    Example:
        >>> from pathlib import Path
        >>> passed = safe_subprocess_run(Path("/tmp/test.py"), cwd="/tmp")
        >>> isinstance(passed, bool)
        True
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
