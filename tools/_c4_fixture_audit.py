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


def _parse_recovering(code: str) -> tuple[ast.Module | None, bool]:
    """Parse *code*; on failure retry on the longest valid line prefix.

    Engine payloads are often mostly-valid Python with a syntactically broken
    tail (truncated generations). Returns (tree, parsed_fully); (None, False)
    when nothing parses. Each retry strictly shrinks the candidate, so the
    loop terminates.
    """
    lines = code.splitlines()
    parsed_fully = True
    while lines:
        try:
            return ast.parse("\n".join(lines)), parsed_fully
        except SyntaxError as e:
            parsed_fully = False
            if e.lineno is not None and 1 <= e.lineno <= len(lines):
                lines = lines[: e.lineno - 1]
            else:
                lines = lines[:-1]
    return None, False


def introduced_symbols(code: str) -> set[str]:
    """Names bound in *code*: def/class names and assignment targets.

    Broken-tail payloads are measured on their longest valid prefix.
    """
    tree, _ = _parse_recovering(code)
    if tree is None:
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
    """Name ids READ (Load context) and attribute names referenced in *code*.

    Broken-tail payloads are measured on their longest valid prefix.
    """
    tree, _ = _parse_recovering(code)
    if tree is None:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        # Store-context Names are rebindings, not reads: `x = 2` after a prior
        # `x = 1` must not count as reuse
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def code_rounds(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(code, action, target) per code-bearing record, in step order."""
    from rune.engine.continuation import extract_partial_code  # noqa: PLC0415

    recs = sorted(
        (r for r in records if r.get("action") in _CODE_ACTIONS),
        key=lambda r: r.get("step", 0),
    )
    out: list[tuple[str, str, str]] = []
    for r in recs:
        code = extract_partial_code(r.get("output") or "")
        if code.strip():
            out.append((code, str(r.get("action") or ""), str(r.get("target") or "")))
    return out


def reuse_counts(
    rounds: list[tuple[str, str, str]],
) -> tuple[int, int, list[dict[str, Any]]]:
    """(reused rounds, eligible rounds t>=2, per-adjacent-pair detail).

    The headline fraction (reused/eligible) keeps its pre-registered pooled
    definition; the pair detail lets the findings doc stratify code->repair
    adjacencies (trivially self-reusing) from cross-subtask code->code pairs.
    """
    reused = eligible = 0
    pairs: list[dict[str, Any]] = []
    parsed = [_parse_recovering(c)[1] for c, _, _ in rounds]
    for i, ((p_code, p_act, p_tgt), (c_code, c_act, c_tgt)) in enumerate(
        zip(rounds, rounds[1:], strict=False)
    ):
        eligible += 1
        hit = bool(introduced_symbols(p_code) & used_symbols(c_code))
        reused += int(hit)
        pairs.append({
            "prev_action": p_act, "curr_action": c_act,
            "prev_target": p_tgt, "curr_target": c_tgt,
            "same_target": p_tgt == c_tgt, "reused": hit,
            "prev_parsed_fully": parsed[i], "curr_parsed_fully": parsed[i + 1],
        })
    return reused, eligible, pairs


def audit_session(path: Path) -> dict[str, Any]:
    """Per-session reuse report for one session.jsonl."""
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rounds = code_rounds(records)
    reused, eligible, pairs = reuse_counts(rounds)
    # Strict = pre-fix instrument: reuse only counted when both payloads parse
    # whole (prefix == payload); a parse failure on either side scored False.
    reused_strict = sum(
        1 for p in pairs
        if p["reused"] and p["prev_parsed_fully"] and p["curr_parsed_fully"]
    )
    return {
        "session": path.parent.name,
        "n_records": len(records),
        "n_code_rounds": len(rounds),
        "eligible_rounds": eligible,
        "reused_rounds": reused,
        "reused_rounds_strict": reused_strict,
        "pairs": pairs,
        "introduced_per_round": [sorted(introduced_symbols(c)) for c, _, _ in rounds],
        "parsed_fully_per_round": [_parse_recovering(c)[1] for c, _, _ in rounds],
    }


def main() -> None:
    """Scan --sessions-dir, print the per-session table, write --out JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    reports = [audit_session(p) for p in sorted(a.sessions_dir.rglob("session.jsonl"))]
    reused = sum(r["reused_rounds"] for r in reports)
    reused_strict = sum(r["reused_rounds_strict"] for r in reports)
    eligible = sum(r["eligible_rounds"] for r in reports)
    frac = (reused / eligible) if eligible else None
    frac_strict = (reused_strict / eligible) if eligible else None
    for r in reports:
        print(
            f"{r['session']}: reuse {r['reused_rounds']}/{r['eligible_rounds']}"
            f" (strict {r['reused_rounds_strict']}, code rounds: {r['n_code_rounds']})"
        )
    tail = (
        f" = {frac:.3f} (strict {reused_strict}/{eligible} = {frac_strict:.3f})"
        if frac is not None and frac_strict is not None
        else "  [no eligible rounds]"
    )
    print(f"TOTAL: {reused}/{eligible}{tail}")
    if a.out:
        a.out.write_text(json.dumps(
            {"sessions": reports, "total_reused_rounds": reused,
             "total_reused_rounds_strict": reused_strict,
             "total_eligible_rounds": eligible, "reuse_fraction": frac,
             "reuse_fraction_strict": frac_strict},
            indent=1,
        ))


if __name__ == "__main__":
    main()
