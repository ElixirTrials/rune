"""Code-state preservation evaluations for the reasoning loop.

Measures how well adapter compression preserves structural properties
of the code artifact across turns.
"""

from __future__ import annotations

import builtins as _builtins_mod
import keyword
import re

_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_]\w*)\b")
_BUILTIN_NAMES = set(dir(_builtins_mod))
_SKIP = _BUILTIN_NAMES | set(keyword.kwlist) | {
    "self", "cls", "None", "True", "False", "return", "pass", "import", "from",
    "class", "def", "if", "else", "elif", "for", "while", "try", "except",
    "finally", "with", "as", "yield", "raise", "break", "continue", "and",
    "or", "not", "in", "is", "lambda", "global", "nonlocal", "assert", "del",
}

_SIG_RE = re.compile(r"^(?:def|class)\s+\w+\s*\([^)]*\)", re.MULTILINE)


def _extract_identifiers(code: str) -> set[str]:
    """Extract non-keyword, non-builtin identifiers from code."""
    ids = set(_IDENTIFIER_RE.findall(code))
    return ids - _SKIP


def compute_identifier_recall(
    previous_code: str,
    current_output: str,
) -> float:
    """Fraction of previous identifiers that appear in current output."""
    prev_ids = _extract_identifiers(previous_code)
    if not prev_ids:
        return 1.0
    if not current_output:
        return 0.0
    curr_ids = _extract_identifiers(current_output)
    recalled = prev_ids & curr_ids
    return len(recalled) / len(prev_ids)


def compute_signature_consistency(
    previous_interfaces: str,
    current_interfaces: str,
) -> float:
    """Fraction of previous function/class signatures unchanged in current."""
    prev_sigs = set(_SIG_RE.findall(previous_interfaces))
    if not prev_sigs:
        prev_lines = {
            line.strip()
            for line in previous_interfaces.splitlines()
            if line.strip()
        }
        if not prev_lines:
            return 1.0
        curr_lines = {
            line.strip()
            for line in current_interfaces.splitlines()
            if line.strip()
        }
        return len(prev_lines & curr_lines) / len(prev_lines)

    curr_sigs = set(_SIG_RE.findall(current_interfaces))
    return len(prev_sigs & curr_sigs) / len(prev_sigs)


def compute_import_preservation(
    previous_imports: str,
    current_code: str,
) -> float:
    """Fraction of original import lines still present in current code."""
    prev_lines = {
        line.strip()
        for line in previous_imports.splitlines()
        if line.strip()
    }
    if not prev_lines:
        return 1.0
    curr_lines = {line.strip() for line in current_code.splitlines() if line.strip()}
    preserved = prev_lines & curr_lines
    return len(preserved) / len(prev_lines)


def compute_regression_reintroduction(
    fixed_test_names: list[str],
    currently_failing: list[str],
) -> float:
    """Fraction of previously-fixed tests that are still passing (not regressed)."""
    if not fixed_test_names:
        return 1.0
    failing_set = set(currently_failing)
    regressed = [t for t in fixed_test_names if t in failing_set]
    return 1.0 - len(regressed) / len(fixed_test_names)
