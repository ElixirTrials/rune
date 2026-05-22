from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


def extract_code(raw: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if blocks:
        return str(max(blocks, key=len)).strip()
    return raw.strip()


def run_in_sandbox(code: str, *, timeout: int = 30) -> ExecutionResult:
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
