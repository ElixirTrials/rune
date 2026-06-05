"""Mine successful repair/solve episodes from engine sessions into a distillation
corpus (REMOVE-BEFORE-MERGE). The RFT / rejection-sampling step of issue #52:

The reward is the in-loop oracle (and, optionally, the held-out result). For every
code/repair step whose generated code PASSED the oracle, emit one corpus row in the
EXISTING distill format ({task_id, entry_point, context, answer}) — context is the
adapter conditioning the engine actually used (the new recall format when the run
was episodic/escalate), answer is the code that passed. The existing distiller
then re-trains the hypernet on these positives. Repair steps that flipped a failure
to a pass are the iterative-logic signal we most want to reinforce.

No new trainer: this only produces the corpus the distill stage already consumes.

Run:
  uv run python tools/_rft_mine.py --sessions DIR [DIR ...] --out corpus.jsonl \
      [--repairs-only] [--held-out-only]
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _entry_point(code: str) -> str:
    """First top-level function name in *code* (the function the episode defines)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return ""


def _mine_session(
    session_dir: Path, *, repairs_only: bool, held_out_only: bool
) -> list[dict]:
    from rune.engine.continuation import extract_partial_code  # noqa: PLC0415

    meta_path = session_dir / "metadata.json"
    sess_path = session_dir / "session.jsonl"
    if not sess_path.exists():
        return []
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    task_id = meta.get("problem_id", session_dir.name)
    if held_out_only and not meta.get("pass_at_1"):
        return []

    steps = [json.loads(line) for line in sess_path.read_text().splitlines() if line.strip()]
    # a step is a positive if its code passed the oracle (exit 0). A repair that
    # followed a failing attempt is the iterative-logic signal.
    prior_failed = False
    rows: list[dict] = []
    for s in steps:
        action = s.get("action")
        fb = s.get("feedback") or {}
        if action in ("code", "repair"):
            passed = fb.get("exit_code") == 0
            is_repair_fix = action == "repair" and prior_failed
            if passed and (is_repair_fix or not repairs_only):
                code = extract_partial_code(s.get("output", "")).strip()
                ctx = s.get("trajectory", "")
                ep = _entry_point(code)
                if code and ctx and ep:
                    rows.append(
                        {
                            "task_id": task_id,
                            "entry_point": ep,
                            "context": ctx,
                            "answer": code,
                            "source_action": action,
                        }
                    )
            prior_failed = not passed
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True, help="session root dir(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--repairs-only",
        action="store_true",
        help="only repairs that fixed a prior failure (pure iterative-logic signal)",
    )
    ap.add_argument(
        "--held-out-only",
        action="store_true",
        help="only tasks whose final code passed the held-out tests (stricter)",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    for root in args.sessions:
        for sess in sorted(Path(root).rglob("session.jsonl")):
            rows.extend(
                _mine_session(
                    sess.parent,
                    repairs_only=args.repairs_only,
                    held_out_only=args.held_out_only,
                )
            )

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    n_repair = sum(r["source_action"] == "repair" for r in rows)
    print(
        f"mined {len(rows)} positive episodes ({n_repair} repairs) -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
