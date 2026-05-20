"""Tests that adapter unload is called after every run_iteration call site."""

from __future__ import annotations

import ast
import re

from pathlib import Path


RUNNER_PATH = Path("scripts/rune_runner.py")


def _get_run_iteration_call_sites() -> list[int]:
    """Find all line numbers where run_iteration() is called."""
    source = RUNNER_PATH.read_text()
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "run_iteration":
                lines.append(node.lineno)
            elif isinstance(func, ast.Attribute) and func.attr == "run_iteration":
                lines.append(node.lineno)
    return sorted(lines)


def _find_unload_after_line(source_lines: list[str], call_line: int) -> bool:
    """Check that unload_adapter or _eager_unload appears within 30 lines after a run_iteration call."""
    start = call_line  # 0-indexed: call_line is 1-indexed
    end = min(start + 30, len(source_lines))
    window = "\n".join(source_lines[start:end])
    return "unload_adapter" in window or "_eager_unload" in window or "eager_unload_fn" in window


def test_every_run_iteration_has_unload() -> None:
    """Every run_iteration() call site must have unload_adapter within 30 lines."""
    source = RUNNER_PATH.read_text()
    source_lines = source.splitlines()
    call_sites = _get_run_iteration_call_sites()

    assert len(call_sites) >= 9, f"Expected >=9 call sites, found {len(call_sites)}"

    missing = []
    for line_no in call_sites:
        if not _find_unload_after_line(source_lines, line_no):
            missing.append(line_no)

    assert not missing, (
        f"run_iteration() call sites missing unload_adapter within 30 lines: "
        f"lines {missing}"
    )


def test_cleanup_phase_adapters_still_exists() -> None:
    """_cleanup_phase_adapters() remains as a safety net."""
    source = RUNNER_PATH.read_text()
    assert "_cleanup_phase_adapters" in source
