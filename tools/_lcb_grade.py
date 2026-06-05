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
from pathlib import Path

sys.path.insert(0, "/tmp/LiveCodeBench")
from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: E402
    codegen_metrics,
)

LCB_JSONL = "/tmp/lcb/test6.jsonl"


def _decode_private(s: str) -> list:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode()))))


def _sample(row: dict) -> dict:
    tc = json.loads(row["public_test_cases"]) + _decode_private(row["private_test_cases"])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--timeout", type=int, default=6)
    args = ap.parse_args()

    rows = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in Path(LCB_JSONL).read_text().splitlines()
    }
    gens = json.loads(Path(args.gens).read_text())

    samples_list = [_sample(rows[g["question_id"]]) for g in gens]
    generations_list = [g["code_list"] for g in gens]
    metrics, _results, _ = codegen_metrics(
        samples_list, generations_list, k_list=[1], num_process_evaluate=8,
        timeout=args.timeout,
    )
    print(f"LCB pass@1 = {metrics.get('pass@1')}  (n={len(gens)})")


if __name__ == "__main__":
    main()
