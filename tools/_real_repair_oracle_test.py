"""Force the REAL engine repair step under perfect vs real-engine oracle.

REMOVE-BEFORE-MERGE. Drives the ACTUAL engine repair path for each false-pass
task: render_episode_adapter -> prompt_episodic_repair -> adapter-ON
(scaling=1.0, c3 ckpt) -> model.generate  -- exactly graph.step_node's repair
branch.

The critique is built + VERIFIED well-formed by tools/_verify_critique.py
(clean observed/expected for WRONG, performance critique for TLE, large inputs
summarized not truncated) and passed intact through both channels (verified by
tools/_verify_passing.py). Grading is IN-PROCESS so the critique uses real
objects, not truncated sandbox reprs.

Only ONE thing varies: which failing case the oracle saw.
  perfect      : first HIDDEN test the shipped code fails (ground truth).
  real_engine  : first PUBLIC test the shipped code fails (what the in-loop
                 oracle actually sees). For false-pass tasks public usually
                 PASSES -> no failing case -> repair never fires (ships wrong).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
FALSE_PASS = [
    "3701",
    "3705",
    "3717",
    "3743",
    "3754",
    "3760",
    "3771",
    "3777",
    "3786",
    "3791",
    "3793",
]

_vc = importlib.util.spec_from_file_location(
    "vc", "/workspaces/content/tools/_verify_critique.py"
)
vc = importlib.util.module_from_spec(_vc)
_vc.loader.exec_module(vc)

_GLOBALS_PREAMBLE = (
    "from typing import *\n"
    "import collections, math, heapq, bisect, itertools, functools, re\n"
    "from collections import defaultdict, deque, Counter, OrderedDict\n"
)


def _load_fn(code: str, entry: str):
    ns: dict[str, Any] = {}
    exec(_GLOBALS_PREAMBLE, ns)  # noqa: S102
    exec(code, ns)  # noqa: S102
    return ns[entry]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/goal3/overnight/real_repair_oracle.json")
    ap.add_argument(
        "--n",
        type=int,
        default=1,
        help="repeats per task for the perfect arm (temp 0.3 is noisy)",
    )
    args = ap.parse_args()

    from rune.bench.lcb import extract_entry_function  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.graph import (  # noqa: PLC0415
        _effective_scaling,
        render_episode_adapter,
        state_to_ctx,
    )
    from rune.engine.parse import extract_code_block, render_template  # noqa: PLC0415
    from rune.engine.policy import ACTIONS  # noqa: PLC0415
    from rune.engine.state import Feedback, StepRecord, Subtask  # noqa: PLC0415
    from rune.model.adapter import apply_episodic_adapter  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    rows = {
        json.loads(x)["question_id"]: json.loads(x)
        for x in Path(LCB).read_text().splitlines()
    }
    cands = {
        g["question_id"]: g["code_list"][0]
        for g in json.loads(Path(COMBINED).read_text())
    }

    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT,
        adapter_scaling=1.0,
        prompt_mode="escalate",
        model_judge=False,
    )
    model = ModelWrapper.from_config(cfg)
    base_repair = ACTIONS["repair"]

    def solved(code: str, entry: str, cases: list) -> bool:
        try:
            fn = _load_fn(code, entry)
        except Exception:  # noqa: BLE001
            return False
        return vc.first_failure(fn, cases) is None

    async def run_repair(
        crit: str, entry: str, spec: str, sig: str, wrong: str, obs_line: str
    ) -> str:
        tried_err = obs_line.replace("observed: ", "")
        state = {
            "entry_point": entry,
            "signature": sig,
            "task": spec,
            "public_checks": "",
            "overall_goal": spec,
            "subtasks": [
                Subtask(
                    name=entry,
                    description="",
                    depends_on=[],
                    acceptance_check="",
                    builds=entry,
                )
            ],
            "code_results": {entry: wrong},
            "best_code": {entry: wrong},
            "feedback": {entry: Feedback(stdout="", stderr=crit, exit_code=1)},
            "diagnosis": {entry: crit},
            "repair_briefs": {entry: crit},
            "plans": {entry: ""},
            "plan_rejections": {},
            "integration_feedback": None,
            "trajectory": [
                StepRecord(
                    step=2,
                    action_name="code",
                    target_subtask=entry,
                    adapter_id=None,
                    feedback=Feedback(stdout="", stderr=tried_err, exit_code=1),
                    generated_code=wrong,
                )
            ],
        }
        act = replace(base_repair, target_subtask=entry)
        ctx = state_to_ctx(state, act)
        traj = render_episode_adapter("repair", entry, state)
        prompt = render_template("prompt_episodic_repair", **ctx)
        scaling = _effective_scaling("escalate", act, state["code_results"], 1.0)
        apply_episodic_adapter(model, traj, scaling=scaling)
        gen = await model.generate(
            prompt=prompt,
            system_prompt=act.system_prompt,
            max_tokens=2048,
            temperature=0.3,
            thinking_budget=0,
        )
        return extract_code_block(gen.text) or ""

    results = []
    for qid in FALSE_PASS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        sig = row.get("starter_code", "") or ""
        import ast  # noqa: PLC0415

        all_cases = vc._cases(row)
        pub_cases = []
        for t in json.loads(row["public_test_cases"]):
            try:
                pub_cases.append(
                    (
                        [
                            ast.literal_eval(x)
                            for x in t["input"].split("\n")
                            if x.strip()
                        ],
                        ast.literal_eval(t["output"]),
                    )
                )
            except (ValueError, SyntaxError):
                continue
        wrong = extract_entry_function(cands.get(qid, ""), entry)
        try:
            fn = _load_fn(wrong, entry)
        except Exception:  # noqa: BLE001
            results.append({"qid": qid, "entry": entry, "load_err": True})
            continue

        hidden = vc.first_failure(fn, all_cases)
        public = vc.first_failure(fn, pub_cases)
        rec = {
            "qid": qid,
            "entry": entry,
            "hidden_kind": hidden[0] if hidden else "NONE",
            "public_fails": public is not None,
        }

        for cond, fail, reps in (
            ("perfect", hidden, args.n),
            ("real_engine", public, 1),
        ):
            if fail is None:
                rec[cond] = {"fired": False, "solved": 0, "reps": reps, "changed": 0}
                continue
            kind, fargs, got, want = fail
            crit = vc.build_critique(entry, kind, fargs, got, want)
            obs_line = next(
                ln for ln in crit.splitlines() if ln.startswith("observed:")
            )
            sv = ch = 0
            for _ in range(reps):
                new = await run_repair(crit, entry, spec, sig, wrong, obs_line)
                new_fn = extract_entry_function(new, entry)
                ch += int(new_fn.strip() != wrong.strip())
                sv += int(solved(new, entry, all_cases) if new_fn.strip() else False)
            rec[cond] = {"fired": True, "changed": ch, "solved": sv, "reps": reps}
        print(
            f"{qid} {entry:26s} hidden={rec['hidden_kind']:12s} "
            f"pubfail={rec['public_fails']!s:5s} "
            f"perfect={rec['perfect']['solved']}/{rec['perfect']['reps']} "
            f"real={rec['real_engine']['solved']}/{rec['real_engine']['reps']}",
            flush=True,
        )
        results.append(rec)

    Path(args.out).write_text(json.dumps(results, indent=2))
    testable = [
        r
        for r in results
        if r.get("hidden_kind") not in (None, "NONE") and not r.get("load_err")
    ]
    wrong_tasks = [r for r in testable if r["hidden_kind"] == "WRONG"]
    tle_tasks = [r for r in testable if r["hidden_kind"] == "TLE"]

    def tally(rs, cond):
        # a task is "recoverable" if perfect solved it at least once across reps
        recov = sum(1 for r in rs if r[cond]["solved"] > 0)
        sv = sum(r[cond]["solved"] for r in rs)
        reps = sum(r[cond]["reps"] for r in rs)
        return recov, sv, reps, len(rs)

    print(
        f"\n=== REAL ENGINE repair path (adapter-on, scaling=1.0), clean critiques, "
        f"N={args.n} ==="
    )
    print(
        f"testable: {len(testable)}/{len(results)} "
        f"(WRONG={len(wrong_tasks)}, TLE={len(tle_tasks)}, "
        f"undecodable={len(results) - len(testable)})"
    )
    for label, rs in (
        ("WRONG (logic)", wrong_tasks),
        ("TLE (perf)", tle_tasks),
        ("ALL testable", testable),
    ):
        recov, sv, reps, n = tally(rs, "perfect")
        print(
            f"  {label:16s}: perfect recovered {recov}/{n} tasks "
            f"(>=1 solve); solve-rate {sv}/{reps}"
        )
    print(
        "  (real-engine arm fires only when public fails; all "
        f"{sum(1 for r in testable if not r['public_fails'])}/{len(testable)} "
        "testable pass public -> repair never fires)"
    )


if __name__ == "__main__":
    asyncio.run(main())
