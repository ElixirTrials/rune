"""Sandbox executor: code extraction and subprocess-based code runner."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    """Result of running code in the sandbox.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Process exit code; 0 indicates success, -1 indicates timeout.
    """

    stdout: str
    stderr: str
    exit_code: int


def extract_code(raw: str) -> str:
    """Extract the longest fenced code block from a raw model response.

    Args:
        raw: Raw text that may contain markdown code fences.

    Returns:
        Content of the longest ```python``` or ``` block, or the stripped
        input if no fences are found.
    """
    blocks = re.findall(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if blocks:
        return str(max(blocks, key=len)).strip()
    return raw.strip()


def run_in_sandbox(code: str, *, timeout: int = 30) -> ExecutionResult:
    """Execute Python code in a temporary file via subprocess.

    Args:
        code: Python source code to run.
        timeout: Maximum seconds to wait before killing the process.

    Returns:
        ExecutionResult with stdout, stderr, and exit code.
        exit_code is -1 on timeout.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        path = Path(f.name)
    try:
        proc = subprocess.run(
            ["python", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ExecutionResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(stdout="", stderr="Timeout", exit_code=-1)
    finally:
        path.unlink(missing_ok=True)
