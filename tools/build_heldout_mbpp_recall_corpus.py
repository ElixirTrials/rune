"""Build held-out MBPP recall corpora (issue #52 Phase-1 generalization gate).

The 10-episode cross-over is TRAINED-ON-TEST. To test whether the recall objective
GENERALIZES (encode ANY body accessibly) vs MEMORIZES the 10, we need disjoint
train/eval splits of fresh MBPP tasks. Emits two jsonl in the committed corpus format
(task_id, entry_point, context, answer) PLUS description/reference/test_code so one
file drives both the accessibility probe and the real-MBPP pass@1 probe.

context == render_training_format_trajectory(task=description) (byte-identical scaffold
to the cross-over corpus); answer/reference == MBPP canonical solution; test_code == the
real 3-assert suite (+imports). Excludes the 10 cross-over tasks and bodyless refs.

Run: uv run python tools/build_heldout_mbpp_recall_corpus.py [--n-train 40 --n-eval 24]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rune.engine.graph import render_training_format_trajectory

CROSSOVER_IDS = {11, 12, 14, 16, 17, 18, 19, 20, 56, 57}
_FUNC = re.compile(r"assert\s+(\w+)\s*\(")


def _entry_point(test_list: list[str]) -> str | None:
    for t in test_list:
        m = _FUNC.search(t)
        if m:
            return m.group(1)
    return None


def _has_body(code: str, entry: str) -> bool:
    """Reference must define `def <entry>(` with a body line after the signature."""
    j = code.find(f"def {entry}(")
    if j < 0:
        return False
    line_end = code.find("\n", j)
    return line_end >= 0 and len(code[line_end + 1 :].strip()) > 0


def _description(prompt: str, first_test: str) -> str:
    # Byte-identical scaffold to the cross-over corpus descriptions.
    return f'"""\n{prompt}\n\n>>> {first_test}\n"""\n'


def _row(rec: dict) -> dict | None:
    test_list = rec.get("test_list") or []
    if not test_list:
        return None
    entry = _entry_point(test_list)
    code = rec.get("code", "")
    if not entry or not _has_body(code, entry):
        return None
    desc = _description(rec["prompt"], test_list[0])
    test_code = "\n".join(list(rec.get("test_imports") or []) + list(test_list))
    return {
        "task_id": f"mbpp/{rec['task_id']}",
        "entry_point": entry,
        "context": render_training_format_trajectory(task=desc),
        "answer": code,
        "description": desc,
        "reference": code,
        "test_code": test_code,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=24)
    a = ap.parse_args()

    import datasets  # noqa: PLC0415

    ds = datasets.load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test"
    )
    rows = []
    for rec in ds:
        if rec["task_id"] in CROSSOVER_IDS:
            continue
        r = _row(rec)
        if r is not None:
            rows.append(r)
    rows.sort(key=lambda r: int(r["task_id"].split("/")[-1]))

    need = a.n_train + a.n_eval
    if len(rows) < need:
        raise SystemExit(f"only {len(rows)} usable held-out tasks, need {need}")
    # Disjoint, deterministic: train/eval are sequential difficulty-matched blocks.
    train = rows[: a.n_train]
    eval_ = rows[a.n_train : a.n_train + a.n_eval]

    out_dir = Path("benchmarks")
    for name, split in (("mbpp_recall_train", train), ("mbpp_recall_heldout", eval_)):
        p = out_dir / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in split) + "\n")
        ids = [r["task_id"] for r in split]
        print(f"[OK] {p}  n={len(split)}  ids={ids}")
    overlap = set(r["task_id"] for r in train) & set(r["task_id"] for r in eval_)
    print(f"[OK] disjoint overlap: {overlap or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
