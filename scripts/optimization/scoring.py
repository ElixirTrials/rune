"""Scoring function for optimization trials.

Evaluates generated code on multiple signals:
execution success, domain relevance, structural quality,
and absence of degenerate patterns.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

from task_pool import EvalTask


@dataclass(frozen=True)
class CodeScore:
    """Multi-signal score for generated code."""

    tests_pass: bool
    domain_hits: int  # 0-5
    structure_hits: int  # 0-3 (import unittest, TestCase, main)
    not_degenerate: bool
    line_count: int
    error: str

    @property
    def total(self) -> float:
        """Weighted composite score (0-11.5 max)."""
        length_norm = min(1.0, max(0.0, (self.line_count - 10) / 70))
        return (
            3.0 * self.tests_pass
            + 1.0 * self.domain_hits
            + 1.0 * self.structure_hits
            + 2.0 * self.not_degenerate
            + 0.5 * length_norm
        )


def extract_code(text: str) -> str:
    """Extract Python from markdown blocks if present."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _check_degenerate(code: str) -> bool:
    """Return True if NOT degenerate (no excessive repetition)."""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False  # too short = degenerate
    unique = len(set(lines))
    return unique > len(lines) * 0.5


def score_code(code: str, task: EvalTask, timeout: int = 15) -> CodeScore:
    """Score generated code against a task."""
    code = extract_code(code)

    # Domain relevance
    code_lower = code.lower()
    domain_hits = sum(1 for kw in task.domain_keywords if kw in code_lower)
    domain_hits = min(domain_hits, 5)

    # Structure
    has_import_ut = "import unittest" in code
    has_testcase = "TestCase" in code
    has_main = "unittest.main" in code
    structure_hits = sum([has_import_ut, has_testcase, has_main])

    # Degenerate check
    not_degenerate = _check_degenerate(code)

    # Execution
    tests_pass = False
    error = ""
    if not_degenerate and code.strip():
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            tests_pass = result.returncode == 0
            if not tests_pass:
                err_lines = result.stderr.strip().splitlines()
                error = err_lines[-1][:120] if err_lines else "unknown error"
        except subprocess.TimeoutExpired:
            error = "timeout"
        except Exception as e:
            error = str(e)[:120]
    elif not not_degenerate:
        error = "degenerate repetition"

    return CodeScore(
        tests_pass=tests_pass,
        domain_hits=domain_hits,
        structure_hits=structure_hits,
        not_degenerate=not_degenerate,
        line_count=len(code.splitlines()),
        error=error,
    )
