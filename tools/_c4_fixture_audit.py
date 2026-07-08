"""C4 I0 — symbol-reuse audit over engine session traces.

For each session.jsonl under --sessions-dir, extract the code payload of every
code-bearing round (actions: code, repair, integrate) and measure the fraction
of rounds t>=2 that reuse at least one symbol introduced in the previous
code-bearing round. The committed lcb_engine_fixes fixtures are step-0
decompose-only, so they report zero eligible rounds; the audit is meaningful
over full regenerated sessions (c4_implementation_plan.md Task 2).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

_CODE_ACTIONS = frozenset({"code", "repair", "integrate"})


def introduced_symbols(code: str) -> set[str]:
    """Names bound in *code*: def/class names and assignment targets."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(
            node.target, ast.Name
        ):
            out.add(node.target.id)
    return out


def used_symbols(code: str) -> set[str]:
    """All Name ids and attribute names referenced in *code*."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def code_rounds(records: list[dict[str, Any]]) -> list[str]:
    """Code payload per code-bearing record, in step order."""
    from rune.engine.continuation import extract_partial_code  # noqa: PLC0415

    recs = sorted(
        (r for r in records if r.get("action") in _CODE_ACTIONS),
        key=lambda r: r.get("step", 0),
    )
    out: list[str] = []
    for r in recs:
        code = extract_partial_code(r.get("output") or "")
        if code.strip():
            out.append(code)
    return out


def reuse_counts(rounds: list[str]) -> tuple[int, int]:
    """(rounds reusing a prev-round-introduced symbol, eligible rounds t>=2)."""
    reused = eligible = 0
    for prev, curr in zip(rounds, rounds[1:], strict=False):
        eligible += 1
        if introduced_symbols(prev) & used_symbols(curr):
            reused += 1
    return reused, eligible


def audit_session(path: Path) -> dict[str, Any]:
    """Per-session reuse report for one session.jsonl."""
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rounds = code_rounds(records)
    reused, eligible = reuse_counts(rounds)
    return {
        "session": path.parent.name,
        "n_records": len(records),
        "n_code_rounds": len(rounds),
        "eligible_rounds": eligible,
        "reused_rounds": reused,
        "introduced_per_round": [sorted(introduced_symbols(c)) for c in rounds],
    }


def main() -> None:
    """Scan --sessions-dir, print the per-session table, write --out JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    reports = [audit_session(p) for p in sorted(a.sessions_dir.rglob("session.jsonl"))]
    reused = sum(r["reused_rounds"] for r in reports)
    eligible = sum(r["eligible_rounds"] for r in reports)
    frac = (reused / eligible) if eligible else None
    for r in reports:
        print(
            f"{r['session']}: reuse {r['reused_rounds']}/{r['eligible_rounds']}"
            f" (code rounds: {r['n_code_rounds']})"
        )
    tail = f" = {frac:.3f}" if frac is not None else "  [no eligible rounds]"
    print(f"TOTAL: {reused}/{eligible}{tail}")
    if a.out:
        a.out.write_text(json.dumps(
            {"sessions": reports, "total_reused_rounds": reused,
             "total_eligible_rounds": eligible, "reuse_fraction": frac},
            indent=1,
        ))


if __name__ == "__main__":
    main()
