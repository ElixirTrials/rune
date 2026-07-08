"""I0 audit: symbol introduction/reuse over engine session traces."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "_c4_fixture_audit.py"
_spec = importlib.util.spec_from_file_location("_c4_fixture_audit", _TOOL)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "lcb_engine_fixes" / "sessions"
)


def _rec(step: int, action: str, output: str) -> dict:
    return {"step": step, "action": action, "target": "",
            "trajectory": "", "prompt": "", "output": output, "feedback": None}


def test_introduced_symbols_defs_classes_assignments() -> None:
    code = "def f(x):\n    y = 1\n    return y\n\nclass C:\n    pass\n\nz = f(2)\n"
    assert audit.introduced_symbols(code) == {"f", "y", "C", "z"}


def test_introduced_symbols_tolerates_syntax_error() -> None:
    assert audit.introduced_symbols("def broken(:") == set()


def test_introduced_symbols_recovers_broken_tail() -> None:
    # Truncated generations leave a valid prefix + mis-indented tail; the
    # prefix must still be measured (pre-fix behavior returned set()).
    code = (
        "def area(r):\n"
        "    pi = 3.14\n"
        "    return pi * r * r\n"
        "  return pi  # unindent does not match any outer indentation level\n"
    )
    assert "area" in audit.introduced_symbols(code)
    assert "pi" in audit.introduced_symbols(code)


def test_parse_recovering_flags_partial_and_full() -> None:
    tree, full = audit._parse_recovering("x = 1\n")
    assert tree is not None and full is True
    tree, full = audit._parse_recovering("x = 1\n  y = (\n")
    assert tree is not None and full is False
    tree, full = audit._parse_recovering("def broken(:")
    assert tree is None and full is False


def test_reuse_detected_across_broken_tail_round() -> None:
    r1 = ("def helper(a):\n    return a + 1\n", "code", "s1")
    r2 = (
        "def solve(xs):\n    return [helper(x) for x in xs]\n  bad_tail = 1\n",
        "code",
        "s2",
    )
    reused, eligible, pairs = audit.reuse_counts([r1, r2])
    assert (reused, eligible) == (1, 1)
    assert pairs[0]["reused"] is True
    assert pairs[0]["prev_parsed_fully"] is True
    assert pairs[0]["curr_parsed_fully"] is False


def test_strict_fraction_reproduces_pre_fix_instrument(tmp_path: Path) -> None:
    # Recovered reuse counts in the headline but NOT in the strict count,
    # which reproduces the pre-fix tool's number over the same denominator.
    p = tmp_path / "s3" / "session.jsonl"
    p.parent.mkdir()
    lines = [
        _rec(0, "code", "```python\ndef area(r):\n    return 3.14 * r * r\n```"),
        _rec(1, "repair", "```python\ndef main():\n    print(area(2))\n  bad_tail = 1\n```"),
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rep = audit.audit_session(p)
    assert (rep["reused_rounds"], rep["eligible_rounds"]) == (1, 1)
    assert rep["reused_rounds_strict"] == 0
    assert rep["parsed_fully_per_round"] == [True, False]


def test_reuse_detected_when_round2_calls_round1_symbol() -> None:
    r1 = ("def helper(a):\n    return a + 1\n", "code", "s1")
    r2 = ("def solve(xs):\n    return [helper(x) for x in xs]\n", "code", "s2")
    reused, eligible, pairs = audit.reuse_counts([r1, r2])
    assert (reused, eligible) == (1, 1)
    assert pairs == [{
        "prev_action": "code", "curr_action": "code",
        "prev_target": "s1", "curr_target": "s2",
        "same_target": False, "reused": True,
        "prev_parsed_fully": True, "curr_parsed_fully": True,
    }]


def test_no_reuse_counts_zero() -> None:
    reused, eligible, pairs = audit.reuse_counts([
        ("def a():\n    return 1\n", "code", "s1"),
        ("def b():\n    return 2\n", "code", "s2"),
    ])
    assert (reused, eligible) == (0, 1)
    assert pairs[0]["reused"] is False


def test_store_rebinding_is_not_reuse() -> None:
    # `result = 2` REBINDS the name without reading it: not construct-valid reuse.
    reused, eligible, _ = audit.reuse_counts([
        ("result = 1\n", "code", "s1"),
        ("result = 2\n", "code", "s2"),
    ])
    assert (reused, eligible) == (0, 1)


def test_load_context_read_counts_as_reuse() -> None:
    reused, eligible, _ = audit.reuse_counts([
        ("result = 1\n", "code", "s1"),
        ("print(result)\n", "code", "s2"),
    ])
    assert (reused, eligible) == (1, 1)


def test_decompose_only_session_has_no_eligible_rounds(tmp_path: Path) -> None:
    p = tmp_path / "s1" / "session.jsonl"
    p.parent.mkdir()
    p.write_text(json.dumps(_rec(0, "decompose", '{"subtasks": []}')) + "\n")
    rep = audit.audit_session(p)
    assert rep["eligible_rounds"] == 0
    assert rep["n_code_rounds"] == 0


def test_multi_round_session_end_to_end(tmp_path: Path) -> None:
    p = tmp_path / "s2" / "session.jsonl"
    p.parent.mkdir()
    lines = [
        _rec(0, "decompose", '{"subtasks": []}'),
        _rec(1, "code", "```python\ndef area(r):\n    return 3.14 * r * r\n```"),
        _rec(2, "repair", "```python\ndef main():\n    print(area(2))\n```"),
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rep = audit.audit_session(p)
    assert rep["n_code_rounds"] == 2
    assert (rep["reused_rounds"], rep["eligible_rounds"]) == (1, 1)
    assert rep["pairs"] == [{
        "prev_action": "code", "curr_action": "repair",
        "prev_target": "", "curr_target": "",
        "same_target": True, "reused": True,
        "prev_parsed_fully": True, "curr_parsed_fully": True,
    }]
    assert rep["reused_rounds_strict"] == 1
    assert rep["parsed_fully_per_round"] == [True, True]


def test_committed_fixtures_are_step0_only() -> None:
    """Regression-documents the I0 discovery: fixtures have zero code rounds."""
    reports = [audit.audit_session(p) for p in sorted(_FIXTURES.rglob("session.jsonl"))]
    assert len(reports) == 6
    assert all(r["eligible_rounds"] == 0 for r in reports)
