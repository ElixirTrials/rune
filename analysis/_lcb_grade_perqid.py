"""Per-qid official grade for LCB gens. Run in lcbenv.
  PYTHONPATH=/workspaces/content/src:/tmp/LiveCodeBench /tmp/lcbenv/bin/python \
    /tmp/goal3/overnight/_lcb_grade_perqid.py --gens <gens.json> --out <perqid.json>
Emits {qid: {status, n_tests, n_pass}} using the SAME check_correctness the
aggregate grader uses, plus the SAME normalize_lcb_submission applied to code.
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
    check_correctness,
)

from rune.bench.lcb import normalize_lcb_submission  # noqa: E402

LCB_JSONL = "/tmp/lcb/test6.jsonl"


def apply_patches() -> None:
    import lcb_runner.evaluation.testing_util as tu  # noqa: PLC0415

    if getattr(tu, "_rune_patched", False):
        return
    orig = tu.grade_call_based

    def safe(*a: object, **k: object) -> object:
        r = orig(*a, **k)
        if r is None:
            return [-4], {"error_code": -4, "error_message": "compile/fn missing"}
        return r

    def quiet(_s: int, _f: object) -> None:
        raise tu.TimeoutException

    tu.grade_call_based = safe  # type: ignore[method-assign]
    tu.timeout_handler = quiet  # type: ignore[method-assign]

    # RAM guard: cap each untrusted candidate at 4GB so a memory-bomb solution
    # raises MemoryError instead of OOM-crashing the 15GB VM (same fix as _lcb_grade.py).
    _orig_guard = tu.reliability_guard

    def _capped(maximum_memory_bytes: object = None) -> object:
        return _orig_guard(maximum_memory_bytes=maximum_memory_bytes or 4 * 1024**3)

    tu.reliability_guard = _capped  # type: ignore[method-assign]
    tu._rune_patched = True  # type: ignore[attr-defined]


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


def _entry(row: dict) -> str:
    meta = json.loads(row["metadata"]) if row.get("metadata") else {}
    return str(meta.get("func_name") or "")


def _classify(res: list) -> str:
    if res and all(r is True for r in res):
        return "pass"
    if -3 in res:
        return "tle"
    if -4 in res:
        return "runtime"
    if -1 in res:
        return "global_timeout"
    if any(r is False for r in res):
        return "wrong"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=6)
    args = ap.parse_args()

    apply_patches()
    rows = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in Path(LCB_JSONL).read_text().splitlines()
    }
    gens = json.loads(Path(args.gens).read_text())

    out: dict[str, dict] = {}
    for g in gens:
        qid = g["question_id"]
        row = rows[qid]
        raw = g["code_list"][0] if g.get("code_list") else ""
        code = normalize_lcb_submission(
            raw, _entry(row), _starter_code=row.get("starter_code", "")
        )
        empty = not code.strip()
        if empty:
            out[qid] = {"status": "empty", "n_tests": 0, "n_pass": 0, "raw_empty": not raw.strip()}
            print(f"{qid}: empty", flush=True)
            continue
        res, _meta = check_correctness(_sample(row), code, timeout=args.timeout, debug=False)
        status = _classify(res)
        n_pass = sum(1 for r in res if r is True)
        out[qid] = {
            "status": status,
            "n_tests": len(res),
            "n_pass": n_pass,
            "raw_empty": not raw.strip(),
        }
        print(f"{qid}: {status} ({n_pass}/{len(res)})", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    n_pass = sum(1 for v in out.values() if v["status"] == "pass")
    print(f"PERQID pass = {n_pass}/{len(out)}", flush=True)


if __name__ == "__main__":
    main()
