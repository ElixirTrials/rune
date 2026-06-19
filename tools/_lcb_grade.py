"""Grade rune's LiveCodeBench generations with the OFFICIAL LCB harness.

REMOVE-BEFORE-MERGE. Run in the isolated lcbenv (datasets-free):
  PYTHONPATH=/tmp/LiveCodeBench /tmp/lcbenv/bin/python tools/_lcb_grade.py \
    --gens /tmp/lcb/gens_scale0.json
Self-contained: decodes test cases + builds the eval sample (no CodeGenerationProblem
import, so no `datasets` dependency) and calls the official codegen_metrics.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/tmp/LiveCodeBench")

from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: E402
    check_correctness,
    codegen_metrics,
)

from rune.bench.lcb import normalize_lcb_submission  # noqa: E402

LCB_JSONL = "/tmp/lcb/test6.jsonl"


def apply_lcb_harness_patches() -> None:
    """Patch upstream LCB harness edge cases before ``codegen_metrics``."""
    import lcb_runner.evaluation.testing_util as testing_util  # noqa: PLC0415

    if getattr(testing_util, "_rune_patched", False):
        return

    original_grade = testing_util.grade_call_based

    def _grade_call_based_safe(*args: object, **kwargs: object) -> object:
        result = original_grade(*args, **kwargs)
        if result is None:
            return [-4], {
                "error_code": -4,
                "error_message": "Compile failed or entry function not found",
            }
        return result

    def _quiet_timeout_handler(_signum: int, _frame: object) -> None:
        raise testing_util.TimeoutException

    testing_util.grade_call_based = _grade_call_based_safe  # type: ignore[method-assign]
    testing_util.timeout_handler = _quiet_timeout_handler  # type: ignore[method-assign]
    testing_util._rune_patched = True  # type: ignore[attr-defined]


def _decode_private(s: str) -> list:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode()))))


def _sample(row: dict) -> dict:
    tc = json.loads(row["public_test_cases"]) + _decode_private(
        row["private_test_cases"]
    )
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    return {
        "input_output": json.dumps(
            {
                "inputs": [t["input"] for t in tc],
                "outputs": [t["output"] for t in tc],
                "fn_name": meta.get("func_name"),
            }
        )
    }


def _entry_point(row: dict) -> str:
    meta = json.loads(row["metadata"]) if row.get("metadata") else {}
    return str(meta.get("func_name") or "")


def _summarize_outcomes(
    samples_list: list[dict],
    generations_list: list[list[str]],
    *,
    timeout: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sample, gen_list in zip(samples_list, generations_list, strict=True):
        code = gen_list[0] if gen_list else ""
        if not code.strip():
            counts["empty"] += 1
            continue
        res, _meta = check_correctness(sample, code, timeout=timeout, debug=False)
        if res and all(r is True for r in res):
            counts["pass"] += 1
        elif -3 in res:
            counts["tle"] += 1
        elif -4 in res:
            counts["runtime"] += 1
        elif -1 in res:
            counts["global_timeout"] += 1
        elif any(r is False for r in res):
            counts["wrong"] += 1
        else:
            counts["other"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--timeout", type=int, default=6)
    ap.add_argument(
        "--breakdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print pass/wrong/runtime/tle counts after grading",
    )
    args = ap.parse_args()

    apply_lcb_harness_patches()

    rows = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in Path(LCB_JSONL).read_text().splitlines()
    }
    gens = json.loads(Path(args.gens).read_text())

    samples_list = [_sample(rows[g["question_id"]]) for g in gens]
    generations_list = [
        [
            normalize_lcb_submission(
                g["code_list"][0],
                _entry_point(rows[g["question_id"]]),
                _starter_code=rows[g["question_id"]].get("starter_code", ""),
            )
        ]
        for g in gens
    ]

    metrics, _results, _ = codegen_metrics(
        samples_list,
        generations_list,
        k_list=[1],
        num_process_evaluate=8,
        timeout=args.timeout,
    )
    pass_at_1 = float(metrics.get("pass@1", 0.0))
    print(f"LCB pass@1 = {pass_at_1}  (n={len(gens)})")
    if args.breakdown:
        counts = _summarize_outcomes(
            samples_list, generations_list, timeout=args.timeout
        )
        print("breakdown: " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))


if __name__ == "__main__":
    main()
