#!/usr/bin/env python3
"""Replay overnight LCB sessions through post-fix decompose/ship/oracle paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from rune.bench.lcb import (
    build_public_assert_checks,
    extract_entry_function,
    normalize_lcb_submission,
)
from rune.bench.runner import BenchTask, resolve_shipped_code
from rune.engine.graph import build_code_probe
from rune.engine.oracle import defines_entry_point
from rune.engine.parse import candidate_quality, parse_output
from rune.engine.policy import select_action
from rune.engine.state import Action, Feedback, make_initial_state
from rune.engine.validity import validate_solution
from rune.sandbox.executor import run_in_sandbox

LCB_JSONL = Path("/tmp/lcb/test6.jsonl")
SESSIONS = Path("/tmp/goal3/overnight/lcb_escalate_sessions")
GENS = Path("/tmp/goal3/overnight/lcb_escalate.json")

FAILED = ["3753", "3754", "3777"]
CONTROLS = ["3748", "3799", "3801"]


def _row(qid: str) -> dict[str, Any]:
    for line in LCB_JSONL.read_text().splitlines():
        r = json.loads(line)
        if r["question_id"] == qid:
            return r
    raise KeyError(qid)


def _task(qid: str) -> BenchTask:
    row = _row(qid)
    fn = json.loads(row["metadata"])["func_name"]
    public = build_public_assert_checks(row)
    desc = row["question_content"]
    if row.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + row["starter_code"]
    return BenchTask(
        task_id=qid,
        description=desc,
        test_code=public,
        entry_point=fn,
        signature=row.get("starter_code", ""),
        public_checks=public,
    )


def _decompose_raw(qid: str) -> str:
    return json.loads((SESSIONS / qid / "session.jsonl").read_text().splitlines()[0])[
        "output"
    ]


def _best_from_session(qid: str, fn: str) -> tuple[str, int]:
    task = _task(qid)
    best = ""
    bq = -1
    for line in (SESSIONS / qid / "session.jsonl").read_text().splitlines():
        o = json.loads(line)
        if o.get("action") not in ("code", "repair", "integrate"):
            continue
        raw = o.get("output", "")
        code = extract_entry_function(raw, fn) if fn else raw
        if not code.strip() or not defines_entry_point(code, fn):
            continue
        if not validate_solution(
            code,
            entry_point=fn,
            signature=task.signature,
            spec=task.description,
            public_checks=task.public_checks,
        ).ok:
            continue
        r = run_in_sandbox(code + "\n\n" + task.public_checks, timeout=10)
        lcb_q = candidate_quality(code, Feedback("", "", r.exit_code))
        if lcb_q >= bq:
            best = code
            bq = lcb_q
    return best, bq


def _gated_integrate(qid: str, state: dict[str, Any]) -> str:
    integrate = _integrate_code(qid)
    if not integrate.strip():
        return ""
    probe, fired, _ = build_code_probe("", integrate, state)
    if fired and run_in_sandbox(probe, timeout=10).exit_code != 0:
        return ""
    return integrate


def _integrate_code(qid: str) -> str:
    for line in reversed((SESSIONS / qid / "session.jsonl").read_text().splitlines()):
        o = json.loads(line)
        if o.get("action") == "integrate":
            return o.get("output", "")
    return ""


def replay_qid(qid: str) -> dict[str, Any]:
    task = _task(qid)
    fn = task.entry_point
    raw = _decompose_raw(qid)
    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    dec = parse_output(
        Action(
            "decompose", "decompose", "prompt_decompose_concise", "", None, False, None
        ),
        raw,
        None,
        state,
    )
    sub_names = [s.name for s in dec["subtasks"]]
    state.update(dec)

    best, bq = _best_from_session(qid, fn)
    integrate = _gated_integrate(qid, state)
    state.update(
        {
            "plans": {fn: "p"},
            "code_results": {fn: best},
            "code_passed": {fn: False},
            "retries": {fn: 4},
            "best_code": {fn: best},
            "best_quality": {fn: bq},
            "integrated_code": integrate,
        }
    )
    policy_end = select_action(state)
    shipped = resolve_shipped_code(state, task)
    bench_ok = (
        run_in_sandbox(shipped + "\n\n" + task.test_code, timeout=10).exit_code == 0
        if shipped.strip()
        else False
    )

    old_ship = next(
        g["code_list"][0]
        for g in json.loads(GENS.read_text())
        if g["question_id"] == qid
    )
    old_norm = normalize_lcb_submission(old_ship, fn)
    old_bench = (
        run_in_sandbox(old_norm + "\n\n" + task.test_code, timeout=10).exit_code == 0
    )

    return {
        "qid": qid,
        "entry": fn,
        "subtasks_after_fix": sub_names,
        "best_lcb_quality": bq,
        "policy_at_end": [a.name for a in policy_end],
        "shipped_len": len(shipped),
        "bench_pass_new": bench_ok,
        "bench_pass_old": old_bench,
        "shipped_changed": shipped.strip() != old_norm.strip(),
    }


def official_grade(qids: list[str], codes: dict[str, str]) -> dict[str, float]:
    payload = [
        {
            "question_id": q,
            "code_list": [
                normalize_lcb_submission(
                    codes[q], json.loads(_row(q)["metadata"])["func_name"]
                )
            ],
        }
        for q in qids
        if q in codes and codes[q].strip()
    ]
    if not payload:
        return {}
    path = Path("/tmp/lcb_replay_gens.json")
    path.write_text(json.dumps(payload))
    env = {**os.environ, "PYTHONPATH": "/tmp/LiveCodeBench:/tmp/rune-lcb-fixes/src"}
    root = Path(__file__).resolve().parents[1]
    for q in qids:
        single = [p for p in payload if p["question_id"] == q]
        sp = Path(f"/tmp/lcb_replay_{q}.json")
        sp.write_text(json.dumps(single))
        subprocess.run(
            [
                "/tmp/lcbenv/bin/python",
                str(root / "tools/_lcb_grade.py"),
                "--gens",
                str(sp),
                "--timeout",
                "6",
                "--no-breakdown",
            ],
            env=env,
            cwd=str(root),
            capture_output=True,
            check=False,
        )
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default=",".join(FAILED + CONTROLS))
    args = ap.parse_args()
    qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    results = [replay_qid(q) for q in qids]
    print(json.dumps(results, indent=2))
    shipped = {}
    for r in results:
        q = r["qid"]
        task = _task(q)
        best, bq = _best_from_session(q, task.entry_point)
        st = make_initial_state(
            task.description, 12, task.entry_point, task.signature, task.public_checks
        )
        integrate = _gated_integrate(q, st)
        ship_state = {
            "best_code": {task.entry_point: best},
            "best_quality": {task.entry_point: bq},
            "integrated_code": integrate,
        }
        code = resolve_shipped_code(ship_state, task)
        if code.strip():
            shipped[q] = code
    print("\n--- summary ---")
    for r in results:
        print(
            f"q{r['qid']}: subtasks={r['subtasks_after_fix']} "
            f"best_q={r['best_lcb_quality']} policy={r['policy_at_end']} "
            f"bench {r['bench_pass_old']}->{r['bench_pass_new']} "
            f"changed={r['shipped_changed']}"
        )


if __name__ == "__main__":
    main()
