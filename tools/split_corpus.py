"""Deterministic family-keyed train/val/test split (issue #49 reviewer).

The corpus is unrolled PR rounds: multiple rows share a source PR. Splitting by
individual row would leak a PR's rounds across splits, so we split by FAMILY
(metadata.source_task_id) via a stable hash — the same family always lands in the
same bucket, and held-out val/test families are never seen in training. This makes
any post-training eval a real generalization test, not a train-fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict


def family_key(rec: dict) -> str:
    md = rec.get("metadata") or {}
    return str(md.get("source_task_id") or rec.get("task_id") or "unknown")


def bucket(key: str, val_pct: int, test_pct: int) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    if h < val_pct:
        return "val"
    if h < val_pct + test_pct:
        return "test"
    return "train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus", default="/tmp/rune-corpus/external_codereview.unrolled.jsonl"
    )
    ap.add_argument("--out-dir", default="/tmp/rune-corpus")
    ap.add_argument("--val-pct", type=int, default=5)
    ap.add_argument("--test-pct", type=int, default=5)
    a = ap.parse_args()

    with open(a.corpus) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    fams: dict[str, set] = defaultdict(set)
    for r in rows:
        fk = family_key(r)
        b = bucket(fk, a.val_pct, a.test_pct)
        splits[b].append(r)
        fams[b].add(fk)

    # leakage assertion: no family in more than one split
    all_fams = [f for s in fams.values() for f in s]
    assert len(all_fams) == len(set(all_fams)), "FAMILY LEAK across splits!"

    paths = {}
    for name, recs in splits.items():
        p = f"{a.out_dir}/external_codereview.{name}.jsonl"
        with open(p, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        paths[name] = p
        print(f"{name}: {len(recs)} rows / {len(fams[name])} families -> {p}")
    print(f"total families: {len(set(all_fams))} (no leak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
