"""Build size-varied MBPP-recall TRAIN corpora for issue #52 goal-2 (corpus scaling).

Holds the EXISTING 24-task eval set FIXED (benchmarks/mbpp_recall_heldout.jsonl) so accessibility /
pass@1 are comparable across train sizes; grows only the train set, disjoint from eval.

Reuses the exact row format + filters of build_heldout_mbpp_recall_corpus.py. Train pool = all usable
sanitized-test rows MINUS the crossover ids MINUS the fixed eval ids; first N taken (deterministic,
difficulty-ordered). Emits benchmarks/mbpp_recall_train_{N}.jsonl for each requested N.

Run: uv run python tools/build_scaling_train_corpora.py --sizes 80,160
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_heldout_mbpp_recall_corpus import (  # type: ignore  # noqa: PLC2701
    CROSSOVER_IDS,
    _row,
)

OUT = Path("benchmarks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="80,160")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]

    eval_ids = {
        json.loads(ln)["task_id"]
        for ln in (OUT / "mbpp_recall_heldout.jsonl").read_text().splitlines()
        if ln.strip()
    }

    import datasets  # noqa: PLC0415

    ds = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for rec in ds:
        if rec["task_id"] in CROSSOVER_IDS:
            continue
        r = _row(rec)
        if r is not None and r["task_id"] not in eval_ids:
            rows.append(r)
    rows.sort(key=lambda r: int(r["task_id"].split("/")[-1]))
    print(f"[pool] usable train rows (disjoint from {len(eval_ids)} eval): {len(rows)}", flush=True)

    for n in sizes:
        if len(rows) < n:
            print(f"[SKIP] N={n}: only {len(rows)} usable rows", flush=True)
            continue
        split = rows[:n]
        p = OUT / f"mbpp_recall_train_{n}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in split) + "\n")
        overlap = {r["task_id"] for r in split} & eval_ids
        print(f"[OK] {p} n={n} overlap_with_eval={overlap or 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
