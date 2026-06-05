"""Reconstruct benchmarks/mbpp_phase0_iter.json from the committed crossover corpus.

The original mbpp_phase0_iter.json (the frozen E1 probe's task list) was gitignored
and lost on instance recycle. But configs/issue52_mbpp_body_crossover.jsonl carries the
SAME 10 episodes (task_id, entry_point, and `context` = the rendered conditioning
surface). The probe re-renders each task's `description` via
render_training_format_trajectory(task=description); the corpus `context` IS that render
with current_code="" and feedback="". So `description` is recoverable by stripping the
fixed scaffold, and we GATE on a byte-exact round-trip (advisor: surface drift would hit
the trained arm only and bias toward a false NULL).

CPU-only (no torch model load). Run: uv run python tools/_reconstruct_mbpp_phase0.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rune.engine.graph import render_training_format_trajectory

CORPUS = Path("configs/issue52_mbpp_body_crossover.jsonl")
OUT = Path("benchmarks/mbpp_phase0_iter.json")

PREFIX = "## Task\n"
SUFFIX = "\n\n## Current Code\n\n\n## Review Feedback\n"


def recover_description(context: str) -> str:
    if not (context.startswith(PREFIX) and context.endswith(SUFFIX)):
        raise ValueError(
            "context does not match the render_training_format_trajectory scaffold; "
            f"prefix_ok={context.startswith(PREFIX)} suffix_ok={context.endswith(SUFFIX)}"
        )
    return context[len(PREFIX) : len(context) - len(SUFFIX)]


def main() -> int:
    rows = [
        json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()
    ]
    out = []
    for r in rows:
        desc = recover_description(r["context"])
        # GATE A: byte-exact round-trip. If this fails the probe would condition the
        # trained adapter on a surface it was never optimized on -> false NULL risk.
        rendered = render_training_format_trajectory(task=desc)
        if rendered != r["context"]:
            raise AssertionError(
                f"round-trip mismatch for {r['task_id']}:\n"
                f"  rendered={rendered!r}\n  context ={r['context']!r}"
            )
        out.append(
            {
                "task_id": r["task_id"],
                "description": desc,
                "entry_point": r["entry_point"],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[OK] round-trip byte-exact for all {len(out)} episodes")
    print(f"[OK] wrote {OUT} (task_ids: {[r['task_id'] for r in out]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
