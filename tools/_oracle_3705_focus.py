"""3705: standard oracle vs perfect oracle, with the SAME fixed communication.

REMOVE-BEFORE-MERGE. 3705 was solved under the perfect oracle (clean critique).
Question: was that the communication fix or the oracle's COVERAGE? Hold the
(verified-clean) repair channel fixed and vary ONLY the oracle:

  standard oracle : the in-loop oracle = the PUBLIC test cases. If they pass,
                    there is no failing case to diagnose -> repair never fires.
  perfect oracle  : the first HIDDEN failing case (ground truth).

Perfect is run N times (temp 0.3 is stochastic) to get a solve RATE, not n=1.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
QID = "3705"
N = 5

_vc = importlib.util.spec_from_file_location(
    "vc", "/workspaces/content/tools/_verify_critique.py")
vc = importlib.util.module_from_spec(_vc)
_vc.loader.exec_module(vc)

_PRE = ("from typing import *\nimport collections, math, heapq, bisect, itertools, "
        "functools, re\nfrom collections import defaultdict, deque, Counter, OrderedDict\n")


def _load_fn(code: str, entry: str):
    ns: dict[str, Any] = {}
    exec(_PRE, ns)  # noqa: S102
    exec(code, ns)  # noqa: S102
    return ns[entry]


async def main() -> None:
    from rune.bench.lcb import extract_entry_function
    from rune.config import load_rune_config
    from rune.engine.graph import (
        _effective_scaling,
        render_episode_adapter,
        state_to_ctx,
    )
    from rune.engine.parse import extract_code_block, render_template
    from rune.engine.policy import ACTIONS
    from rune.engine.state import Feedback, StepRecord, Subtask
    from rune.model.adapter import apply_episodic_adapter
    from rune.model.wrapper import ModelWrapper

    rows = {json.loads(x)["question_id"]: json.loads(x)
            for x in Path(LCB).read_text().splitlines()}
    cands = {g["question_id"]: g["code_list"][0]
             for g in json.loads(Path(COMBINED).read_text())}
    row = rows[QID]
    meta = json.loads(row["metadata"]) if row.get("metadata") else {}
    entry = meta.get("func_name") or ""
    spec = row.get("question_content", "")[:3500]
    sig = row.get("starter_code", "") or ""
    wrong = extract_entry_function(cands[QID], entry)
    all_cases = vc._cases(row)
    import ast
    pub_cases = []
    for t in json.loads(row["public_test_cases"]):
        try:
            pub_cases.append(([ast.literal_eval(x) for x in t["input"].split("\n")
                               if x.strip()], ast.literal_eval(t["output"])))
        except (ValueError, SyntaxError):
            continue

    fn = _load_fn(wrong, entry)
    pub_fail = vc.first_failure(fn, pub_cases)
    hidden_fail = vc.first_failure(fn, all_cases)

    print(f"# {QID} {entry}: {len(pub_cases)} public cases, {len(all_cases)} total")
    print(f"# STANDARD oracle (public tests) verdict on shipped code: "
          f"{'FAIL '+str(pub_fail[0]) if pub_fail else 'ALL PASS'}")
    print(f"# PERFECT oracle first hidden failure: {hidden_fail[0]}  "
          f"{vc.build_critique(entry, *hidden_fail).splitlines()[1][:90]}")

    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT, adapter_scaling=1.0, prompt_mode="escalate",
        model_judge=False)
    model = ModelWrapper.from_config(cfg)
    base_repair = ACTIONS["repair"]

    async def repair_once(crit: str, obs_line: str) -> tuple[bool, bool]:
        tried = obs_line.replace("observed: ", "")
        state = {
            "entry_point": entry, "signature": sig, "task": spec, "public_checks": "",
            "overall_goal": spec,
            "subtasks": [Subtask(name=entry, description="", depends_on=[],
                                 acceptance_check="", builds=entry)],
            "code_results": {entry: wrong}, "best_code": {entry: wrong},
            "feedback": {entry: Feedback(stdout="", stderr=crit, exit_code=1)},
            "diagnosis": {entry: crit}, "repair_briefs": {entry: crit},
            "plans": {entry: ""}, "plan_rejections": {}, "integration_feedback": None,
            "trajectory": [StepRecord(step=2, action_name="code", target_subtask=entry,
                                      adapter_id=None,
                                      feedback=Feedback(stdout="", stderr=tried,
                                                        exit_code=1),
                                      generated_code=wrong)],
        }
        act = replace(base_repair, target_subtask=entry)
        ctx = state_to_ctx(state, act)
        traj = render_episode_adapter("repair", entry, state)
        prompt = render_template("prompt_episodic_repair", **ctx)
        scaling = _effective_scaling("escalate", act, state["code_results"], 1.0)
        apply_episodic_adapter(model, traj, scaling=scaling)
        gen = await model.generate(prompt=prompt, system_prompt=act.system_prompt,
                                   max_tokens=2048, temperature=0.3, thinking_budget=0)
        new = extract_entry_function(extract_code_block(gen.text) or "", entry)
        if not new.strip():
            return False, False
        return new.strip() != wrong.strip(), solved(new)

    def solved(code: str) -> bool:
        try:
            return vc.first_failure(_load_fn(code, entry), all_cases) is None
        except Exception:  # noqa: BLE001
            return False

    # PERFECT oracle x N
    pcrit = vc.build_critique(entry, *hidden_fail)
    pobs = next(ln for ln in pcrit.splitlines() if ln.startswith("observed:"))
    p_solved = p_changed = 0
    for i in range(N):
        ch, sv = await repair_once(pcrit, pobs)
        p_changed += ch
        p_solved += sv
        print(f"  perfect run {i+1}/{N}: changed={ch} solved={sv}", flush=True)

    # STANDARD oracle: only fires if a PUBLIC case fails
    if pub_fail is None:
        print("\n# STANDARD oracle: public tests all PASS -> diagnose/repair never "
              "fires -> shipped code is the wrong code, UNCHANGED. solved=0/1 (no "
              "repair possible).")
        s_solved = "n/a (repair never fires)"
    else:
        scrit = vc.build_critique(entry, *pub_fail)
        sobs = next(ln for ln in scrit.splitlines() if ln.startswith("observed:"))
        s_solved = 0
        for i in range(N):
            _ch, sv = await repair_once(scrit, sobs)
            s_solved += sv
        s_solved = f"{s_solved}/{N}"

    print(f"\n=== 3705, fixed communication, oracle swapped ===")
    print(f"PERFECT oracle  (sees hidden [0]*50 case): solved {p_solved}/{N} "
          f"(changed {p_changed}/{N})")
    print(f"STANDARD oracle (public tests only):       solved {s_solved}")
    print("\nSame model, same verified-clean repair channel. The only difference is "
          "whether the oracle SAW the failing case -> isolates oracle COVERAGE.")


if __name__ == "__main__":
    asyncio.run(main())
