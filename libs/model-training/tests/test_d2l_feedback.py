"""Tests for heuristic feedback extraction (paper §3.1)."""

from __future__ import annotations

from model_training.d2l_feedback import (
    extract_failure_summary,
    truncate_head_tail,
)
from model_training.d2l_models import Anchor, FeedbackKind


def test_truncate_head_tail_short_passthrough() -> None:
    assert truncate_head_tail("abc", max_bytes=4096) == "abc"


def test_truncate_head_tail_keeps_head_and_tail() -> None:
    body = "H" * 1000 + "M" * 5000 + "T" * 1000
    out = truncate_head_tail(body, max_bytes=4096)
    assert out.startswith("H" * 1000)
    assert out.endswith("T" * 1000)
    assert "[... 2904 bytes elided ...]" in out
    assert len(out.encode("utf-8")) <= 4096 + 64  # marker overhead


def test_extract_pytest_failure() -> None:
    raw = (
        "============================= test session starts ==============================\n"
        "tests/test_foo.py::test_bar FAILED                                       [ 50%]\n"
        "tests/test_foo.py::test_baz PASSED                                       [100%]\n"
        "=================================== FAILURES ===================================\n"
        "FAILED tests/test_foo.py::test_bar - AssertionError: expected 1 got 2\n"
    )
    summary, anchor, kind = extract_failure_summary(raw, hint=None)
    assert kind is FeedbackKind.test_failure
    assert "tests/test_foo.py::test_bar" in summary
    assert "AssertionError" in summary
    assert anchor == Anchor(test="tests/test_foo.py::test_bar")


def test_extract_jest_failure() -> None:
    raw = (
        "FAIL src/foo.test.ts\n"
        "  ● MyComponent › renders\n"
        "    expect(value).toBe(2)\n"
        "    Expected: 2\n"
        "    Received: 3\n"
    )
    summary, anchor, kind = extract_failure_summary(raw, hint=None)
    assert kind is FeedbackKind.test_failure
    assert "src/foo.test.ts" in summary
    assert anchor == Anchor(file="src/foo.test.ts")


def test_extract_lint_summary_with_hint() -> None:
    raw = (
        "src/foo.py:42:5: F401 'os' imported but unused\n"
        "src/foo.py:43:1: E302 expected 2 blank lines\n"
    )
    summary, anchor, kind = extract_failure_summary(raw, hint="lint")
    assert kind is FeedbackKind.lint
    assert "F401" in summary
    assert anchor == Anchor(file="src/foo.py", line=42)


def test_extract_falls_back_to_first_line_for_unknown() -> None:
    raw = "ld: symbol(s) not found for architecture x86_64\nclang: error: linker command failed"
    summary, _anchor, kind = extract_failure_summary(raw, hint="build")
    assert kind is FeedbackKind.build_failure
    assert summary.startswith("ld: symbol(s) not found")
