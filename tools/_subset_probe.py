"""Subset validation probe: do the P0-P2 fixes beat clean24 on a curated subset?

REMOVE-BEFORE-MERGE. Runs the REAL runner (new code) on a curated subset at
proper budget, official-grades each, and diffs per-qid against the clean24
baseline. Subset = regression controls (clean24 PASS, must stay PASS) +
over-cap failing candidates (big specs the 1200->4096 cap-raise + Mission-spec
fix should help).

Usage: this just prints the subset; generation is done by the wrapper shell that
calls _lcb_run.py --qids <subset>. Grading/diff is done here from the produced
generations json + the clean24 sessions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/LiveCodeBench")
sys.path.insert(0, "/workspaces/content/src")

LCB = "/tmp/lcb/test6.jsonl"
CLEAN24_SESSIONS = "/tmp/goal3/overnight/lcb_clean24_sessions"

# Regression controls: clean24 PASS -> must NOT regress.
CONTROLS = ["3832", "3793", "3768", "3809", "3709", "3736"]
# Over-cap failing candidates the fixes target (big specs / known ramble).
CANDIDATES = ["3795", "3701", "3733", "3744", "3743", "3777", "3705", "3760"]
SUBSET = CONTROLS + CANDIDATES


def _grade(gens_path: str) -> dict[str, bool]:
    spec = importlib.util.spec_from_file_location(
        "lcbg", "/workspaces/content/tools/_lcb_grade.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.apply_lcb_harness_patches()
    from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: PLC0415
        check_correctness,  # noqa: PLC0415
    )

    from rune.bench.lcb import (  # noqa: PLC0415
        extract_entry_function,
        normalize_lcb_submission,
    )

    rows = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in Path(LCB).read_text().splitlines()
    }
    out: dict[str, bool] = {}
    for g in json.loads(Path(gens_path).read_text()):
        qid = g["question_id"]
        row = rows[qid]
        entry = (json.loads(row["metadata"]) if row.get("metadata") else {}).get(
            "func_name", ""
        )
        fn = extract_entry_function(g["code_list"][0], entry)
        if not fn.strip():
            out[qid] = False
            continue
        norm = normalize_lcb_submission(
            fn, entry, _starter_code=row.get("starter_code", "")
        )
        res, _ = check_correctness(m._sample(row), norm, timeout=6, debug=False)
        out[qid] = bool(res) and all(r is True for r in res)
    return out


def _clean24_grade(qids: list[str]) -> dict[str, bool]:
    spec = importlib.util.spec_from_file_location(
        "lcbg", "/workspaces/content/tools/_lcb_grade.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.apply_lcb_harness_patches()
    from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: PLC0415
        check_correctness,  # noqa: PLC0415
    )

    from rune.bench.lcb import (  # noqa: PLC0415
        extract_entry_function,
        normalize_lcb_submission,
    )

    rows = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in Path(LCB).read_text().splitlines()
    }
    out: dict[str, bool] = {}
    for qid in qids:
        s = os.path.join(CLEAN24_SESSIONS, qid, "session.jsonl")
        if not os.path.exists(s):
            out[qid] = None  # not run in clean24
            continue
        row = rows[qid]
        entry = (json.loads(row["metadata"]) if row.get("metadata") else {}).get(
            "func_name", ""
        )
        code = ""
        for r in (json.loads(line) for line in Path(s).read_text().splitlines()):
            if (
                r.get("action") in ("code", "repair", "integrate")
                and (r.get("output") or "").strip()
            ):
                code = r["output"]
        fn = extract_entry_function(code, entry)
        if not fn.strip():
            out[qid] = False
            continue
        norm = normalize_lcb_submission(
            fn, entry, _starter_code=row.get("starter_code", "")
        )
        res, _ = check_correctness(m._sample(row), norm, timeout=6, debug=False)
        out[qid] = bool(res) and all(r is True for r in res)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-qids", action="store_true")
    ap.add_argument("--diff", default="", help="path to new-code generations json")
    args = ap.parse_args()

    if args.print_qids:
        print(",".join(SUBSET))
        return

    new = _grade(args.diff)
    old = _clean24_grade(SUBSET)
    print(f"{'qid':6s} {'role':10s} {'clean24':9s} {'newcode':9s} delta")
    regressions, gains = [], []
    for qid in SUBSET:
        role = "control" if qid in CONTROLS else "candidate"
        o = old.get(qid)
        n = new.get(qid)
        os_ = "PASS" if o else ("FAIL" if o is False else "n/a")
        ns_ = "PASS" if n else ("FAIL" if n is False else "n/a")
        delta = ""
        if o and not n:
            delta = "*** REGRESSION ***"
            regressions.append(qid)
        elif (o is False) and n:
            delta = "+++ GAIN +++"
            gains.append(qid)
        print(f"{qid:6s} {role:10s} {os_:9s} {ns_:9s} {delta}")
    no = sum(1 for q in SUBSET if old.get(q))
    nn = sum(1 for q in SUBSET if new.get(q))
    print(
        f"\nclean24 subset pass: {no}/{len(SUBSET)} | newcode subset pass: {nn}/{len(SUBSET)}"  # noqa: E501
    )
    print(f"GAINS: {gains}  REGRESSIONS: {regressions}")
    print(
        "VERDICT:",
        "PASS (>= baseline, no regressions)"
        if (nn >= no and not regressions)
        else "REVIEW (regression or drop)",
    )


if __name__ == "__main__":
    main()
