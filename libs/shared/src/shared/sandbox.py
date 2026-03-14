"""Sandbox execution backends for running untrusted code.

Provides SubprocessBackend (default) and NsjailBackend (Linux-only) for
executing Python code in isolated environments with resource limits.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxResult:
    """Result of executing code in a sandbox.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Process exit code (0 = success).
        timed_out: Whether execution was terminated due to timeout.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class SandboxBackend(ABC):
    """Abstract base class for sandbox execution backends."""

    @abstractmethod
    def run(self, code: str, timeout: int = 30) -> SandboxResult:
        """Execute Python code and return the result.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            A SandboxResult with captured output and exit status.
        """


class SubprocessBackend(SandboxBackend):
    """Execute code via a standard subprocess.

    Writes code to a temporary file and runs it with the current Python
    interpreter. This is the default backend, used on all platforms.
    """

    def run(self, code: str, timeout: int = 30) -> SandboxResult:
        """Execute Python code in a subprocess.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            A SandboxResult with captured output and exit status.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "solution.py")
            with open(script_path, "w") as f:
                f.write(code)

            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )
                return SandboxResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    exit_code=1,
                    timed_out=True,
                )


class NsjailBackend(SandboxBackend):
    """Execute code inside an nsjail sandbox (Linux-only).

    Falls back to SubprocessBackend if nsjail is not found on PATH.
    Applies resource limits via nsjail flags: --time_limit, --rlimit_as,
    --rlimit_cpu.
    """

    def __init__(self) -> None:
        """Initialize NsjailBackend, checking nsjail availability."""
        self._nsjail_path = shutil.which("nsjail")
        self._fallback = SubprocessBackend()
        if self._nsjail_path is None:
            logger.warning(
                "nsjail not found on PATH; NsjailBackend will fall back to subprocess"
            )

    def run(self, code: str, timeout: int = 30) -> SandboxResult:
        """Execute Python code inside nsjail, or fall back to subprocess.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            A SandboxResult with captured output and exit status.
        """
        if self._nsjail_path is None:
            return self._fallback.run(code, timeout)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "solution.py")
            with open(script_path, "w") as f:
                f.write(code)

            cmd = [
                self._nsjail_path,
                "--mode",
                "o",
                "--time_limit",
                str(timeout),
                "--rlimit_as",
                "512",
                "--rlimit_cpu",
                str(timeout),
                "--",
                sys.executable,
                script_path,
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5,
                    cwd=tmpdir,
                )
                return SandboxResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    exit_code=1,
                    timed_out=True,
                )


def get_sandbox_backend() -> SandboxBackend:
    """Return the appropriate sandbox backend based on environment.

    Reads RUNE_EXEC_BACKEND env var:
    - "nsjail" → NsjailBackend (falls back to subprocess if nsjail not found)
    - anything else or unset → SubprocessBackend

    Returns:
        A SandboxBackend instance.
    """
    backend = os.environ.get("RUNE_EXEC_BACKEND", "subprocess")
    if backend == "nsjail":
        return NsjailBackend()
    return SubprocessBackend()
