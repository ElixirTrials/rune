"""Shared continuation utilities for code extraction and quality."""

from __future__ import annotations

import re

from rune.engine.json_repair import extract_code_value
from rune.engine.parse import CodeResult


def extract_partial_code(raw: str) -> str:
    """Extract code from a possibly-truncated CodeResult JSON string.

    Falls back to *raw* when input isn't JSON at all (e.g. continuation
    rounds that emit plain Python).
    """
    try:
        return CodeResult.model_validate_json(raw).code
    except Exception:
        return extract_code_value(raw) or raw


_NUM_RE = re.compile(r"\b\d+\b")


def degeneration_score(text: str, n: int = 4) -> float:
    normalized = _NUM_RE.sub("<N>", text)
    words = normalized.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def validate_syntax(code: str, *, language: str = "python") -> bool:
    if not code or not code.strip():
        return False
    if language != "python":
        raise NotImplementedError(f"syntax validation not implemented for {language!r}")
    try:
        compile(code, "<check>", "exec")
        return True
    except SyntaxError:
        return False
