"""Sandbox executor: subprocess-based code runner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Per-process address-space cap for untrusted code: a runaway/large-input
# solution raises MemoryError instead of OOM-killing the host (the ~15GB box
# crashed twice on unbounded grading). 4GB is far above any benchmark solution.
# Prepended to the code (thread-safe) rather than a preexec_fn, which forks in
# the engine's asyncio worker threads.
_MEM_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
_MEM_GUARD_PROLOGUE = (
    "import resource as _rune_res\n"
    "try:\n"
    f"    _rune_res.setrlimit(_rune_res.RLIMIT_AS, ({_MEM_LIMIT_BYTES}, {_MEM_LIMIT_BYTES}))\n"
    "except Exception:\n"
    "    pass\n"
)


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
        f.write(_MEM_GUARD_PROLOGUE + code)
        f.flush()
        path = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            # Untrusted generated code may emit non-UTF8 bytes; replace rather
            # than raise UnicodeDecodeError out of run_in_sandbox.
            errors="replace",
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
    except OSError as exc:
        # Honour the docstring guarantee: always return an ExecutionResult so a
        # missing interpreter / spawn failure degrades to failed execution
        # instead of aborting the engine step.
        return ExecutionResult(stdout="", stderr=f"sandbox error: {exc}", exit_code=-1)
    finally:
        path.unlink(missing_ok=True)
