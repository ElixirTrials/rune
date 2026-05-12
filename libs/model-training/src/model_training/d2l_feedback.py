"""Heuristic extraction of one-line failure summaries from CI/test/lint output.

The paper defines feedback as raw execution output **plus** "a brief diagnostic
reflection summarising what failed and why, written by … the test harness"
(§3.1). This module produces that reflection deterministically by parsing the
test harness's own structured output (pytest's ``FAILED ... - <reason>`` line,
jest's ``FAIL <file>`` header, lint warning lines, etc.). No LLM, no API cost.
"""

from __future__ import annotations

import re

from model_training.d2l_models import Anchor, FeedbackKind

__all__ = ["extract_failure_summary", "truncate_head_tail"]


_PYTEST_FAILED_RE = re.compile(r"^FAILED\s+(\S+)(?:\s+-\s+(.+))?", re.MULTILINE)
_JEST_FAIL_RE = re.compile(r"^FAIL\s+(\S+)", re.MULTILINE)
_LINT_LINE_RE = re.compile(r"^([^\s:]+):(\d+):\d*:?\s*(\w+)\s*(.*)", re.MULTILINE)


def truncate_head_tail(body: str, max_bytes: int = 4096) -> str:
    """Return ``body`` if it fits in ``max_bytes``; else head+marker+tail."""
    encoded = body.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return body
    half = max_bytes // 2
    head = encoded[:half].decode("utf-8", errors="replace")
    tail = encoded[-half:].decode("utf-8", errors="replace")
    elided = len(encoded) - 2 * half
    return f"{head}\n[... {elided} bytes elided ...]\n{tail}"


def extract_failure_summary(
    raw: str,
    hint: str | None,
) -> tuple[str, Anchor | None, FeedbackKind]:
    """Extract a one-line summary, anchor, and feedback kind from raw output."""
    pytest_match = _PYTEST_FAILED_RE.search(raw)
    if pytest_match:
        test_id = pytest_match.group(1)
        reason = pytest_match.group(2) or ""
        summary = f"{test_id} - {reason}".strip(" -")
        return summary, Anchor(test=test_id), FeedbackKind.test_failure

    jest_match = _JEST_FAIL_RE.search(raw)
    if jest_match:
        path = jest_match.group(1)
        return f"FAIL {path}", Anchor(file=path), FeedbackKind.test_failure

    if hint == "lint":
        lint_match = _LINT_LINE_RE.search(raw)
        if lint_match:
            file, line_str, code, msg = lint_match.groups()
            summary = f"{code} {msg}".strip()
            return (
                summary,
                Anchor(file=file, line=int(line_str)),
                FeedbackKind.lint,
            )

    # Fallback: first non-blank line of raw output.
    first = next((line for line in raw.splitlines() if line.strip()), "")
    kind_map = {
        "lint": FeedbackKind.lint,
        "build": FeedbackKind.build_failure,
    }
    kind = kind_map.get(hint or "", FeedbackKind.ci_failure)
    return first[:200], None, kind
