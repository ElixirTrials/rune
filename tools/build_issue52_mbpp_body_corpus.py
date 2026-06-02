"""Build the 10-row MBPP BODY cross-over corpus (issue #52).

⚠️ REMOVE BEFORE MERGE — trained-on-test cross-over trainability-probe scaffolding.
Depends on the gitignored ``tools/_specificity_probe.py`` (REFS) and
``benchmarks/mbpp_phase0_iter.json`` (both snapshotted on the
``scratch/issue52-research-tools`` remote branch), so it does NOT run from a clean
clone. The committed ``configs/issue52_mbpp_body_crossover.jsonl`` is the canonical
input for reproduction. See the handoff "REMOVE-BEFORE-MERGE manifest".

Produces a JSONL compatible with ``rune.training.hypernet_distill``'s synthetic schema:
each row is ``{"context": ..., "answer": ..., "task_id": ..., "entry_point": ...}``.

This is the trained-on-test cross-over control corpus
(see docs/issue52-predeclared-spec...):
it is *not* intended to measure generalization; it is a trainability probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNE = Path("/workspaces/rune-gpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="/tmp/rune-corpus/issue52_mbpp_body_crossover.jsonl",
        help="Output JSONL path.",
    )
    args = ap.parse_args()

    # Tool-local imports (keeps CPU import surface small; matches repo conventions).
    from rune.engine.graph import render_training_format_trajectory  # noqa: PLC0415

    tasks = json.loads((RUNE / "benchmarks/mbpp_phase0_iter.json").read_text())

    # Reuse the frozen reference solutions from the scoring harness so the corpus
    # can't silently drift from what _specificity_probe.py scores.
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(RUNE / "tools"))
    from _specificity_probe import REFS  # type: ignore  # noqa: PLC0415

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for t in tasks:
            task_id = str(t["task_id"])
            if task_id not in REFS:
                raise KeyError(f"Missing reference solution for {task_id}")
            context = render_training_format_trajectory(task=str(t["description"]))
            rec = {
                "task_id": task_id,
                "entry_point": str(t["entry_point"]),
                "context": context,
                "answer": REFS[task_id],
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

