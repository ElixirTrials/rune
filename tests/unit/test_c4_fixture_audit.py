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


def test_reuse_detected_when_round2_calls_round1_symbol() -> None:
    r1 = "def helper(a):\n    return a + 1\n"
    r2 = "def solve(xs):\n    return [helper(x) for x in xs]\n"
    reused, eligible = audit.reuse_counts([r1, r2])
    assert (reused, eligible) == (1, 1)


def test_no_reuse_counts_zero() -> None:
    reused, eligible = audit.reuse_counts(
        ["def a():\n    return 1\n", "def b():\n    return 2\n"]
    )
    assert (reused, eligible) == (0, 1)


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


def test_committed_fixtures_are_step0_only() -> None:
    """Regression-documents the I0 discovery: fixtures have zero code rounds."""
    reports = [audit.audit_session(p) for p in sorted(_FIXTURES.rglob("session.jsonl"))]
    assert len(reports) == 6
    assert all(r["eligible_rounds"] == 0 for r in reports)
