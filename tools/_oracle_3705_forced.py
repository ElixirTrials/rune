"""3705: FORCE the repair step under three oracle-signal levels (same channel).

REMOVE-BEFORE-MERGE. The standard oracle never fires on 3705 (public passes).
That conflates two failure modes. This FORCES the repair step in all arms and
varies ONLY the information content of the critique, to separate them:

  perfect          : the specific hidden failing case ([0]*50 -> 0).
  standard_generic : "passes all public tests but is wrong on a hidden case; find
                     it" -- knows it is wrong, but has NO specific case.
  standard_silent  : empty critique -- what the public oracle actually yields when
                     public passes (repair fired, no failing case at all).

N runs each (temp 0.3). Isolates: does firing alone help, and does the SPECIFIC
failing case matter beyond just firing?
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
    fn = _load_fn(wrong, entry)
    hidden_fail = vc.first_failure(fn, all_cases)

    perfect_crit = vc.build_critique(entry, *hidden_fail)
    generic_crit = (
        "failure_class: hidden_failure\n"
        "observed: passes all public example tests, but is INCORRECT on at least one "
        "hidden test\n"
        "fix_directive: review the algorithm for missed edge cases (e.g. boundary / "
        "degenerate inputs) and fix it.")

    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT, adapter_scaling=1.0, prompt_mode="escalate",
        model_judge=False)
    model = ModelWrapper.from_config(cfg)
    base_repair = ACTIONS["repair"]

    def solved(code: str) -> bool:
        try:
            return vc.first_failure(_load_fn(code, entry), all_cases) is None
        except Exception:  # noqa: BLE001
            return False

    async def forced_repair(crit: str) -> tuple[bool, bool]:
        # crit == "" -> standard_silent: empty diagnosis/brief, repair still fires.
        obs = next((ln for ln in crit.splitlines() if ln.startswith("observed:")), "")
        # never empty: the engine's adapter renderer indexes stderr.splitlines()[-1]
        tried = obs.replace("observed: ", "") if obs else "incorrect on a hidden test"
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

    arms = [("perfect", perfect_crit), ("standard_generic", generic_crit),
            ("standard_silent", "")]
    summary = {}
    for name, crit in arms:
        ch = sv = 0
        for i in range(N):
            c, s = await forced_repair(crit)
            ch += c
            sv += s
            print(f"  {name:16s} run {i+1}/{N}: changed={c} solved={s}", flush=True)
        summary[name] = (sv, ch)

    print(f"\n=== 3705: FORCED repair, same channel, varying ONLY critique info ===")
    for name, _ in arms:
        sv, ch = summary[name]
        print(f"  {name:16s}: solved {sv}/{N}  (changed {ch}/{N})")
    print("\nperfect = specific hidden case | standard_generic = told-wrong-but-no-case "
          "| standard_silent = repair fired with no failing case at all")


if __name__ == "__main__":
    asyncio.run(main())
