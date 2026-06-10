"""Why is the perfect-oracle prize small? Is the repair note being USED?

REMOVE-BEFORE-MERGE. Perfect oracle recovers only 1/10 tasks. Two explanations:
  (A) passing/attention: the model does not act on the cited failing case.
  (B) capability/teaching: the model FIXES the cited case but cannot generalize
      one example into the correct algorithm.

Discriminator: after a perfect-oracle repair (input X -> expected E), check
whether the new code returns E on X (fixed_given_case) vs passes ALL cases
(solved). fixed_given_case >> solved  => (B). fixed_given_case ~ 0 => (A).
Dumps one real generation per task to eyeball engagement.
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
WRONG_TASKS = ["3701", "3705", "3717", "3743", "3786", "3793"]
N = 3

_vc = importlib.util.spec_from_file_location(
    "vc", "/workspaces/content/tools/_verify_critique.py")
vc = importlib.util.module_from_spec(_vc)
_vc.loader.exec_module(vc)

_PRE = ("from typing import *\nimport collections, math, heapq, bisect, itertools, "
        "functools, re\nfrom collections import defaultdict, deque, Counter, OrderedDict\n")


def _load(code: str, entry: str):
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
    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT, adapter_scaling=1.0, prompt_mode="escalate",
        model_judge=False)
    model = ModelWrapper.from_config(cfg)
    base_repair = ACTIONS["repair"]

    async def repair(crit, entry, spec, sig, wrong, obs) -> str:
        tried = obs.replace("observed: ", "")
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
        sc = _effective_scaling("escalate", act, state["code_results"], 1.0)
        apply_episodic_adapter(model, traj, scaling=sc)
        gen = await model.generate(prompt=prompt, system_prompt=act.system_prompt,
                                   max_tokens=8192, temperature=0.3, thinking_budget=0)
        if gen.truncated:
            print("    [TRUNCATED at 8192 tokens]", flush=True)
        return extract_code_block(gen.text) or ""

    for qid in WRONG_TASKS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        sig = row.get("starter_code", "") or ""
        all_cases = vc._cases(row)
        wrong = extract_entry_function(cands[qid], entry)
        fail = vc.first_failure(_load(wrong, entry), all_cases)
        kind, X, got, want = fail
        crit = vc.build_critique(entry, kind, X, got, want)
        obs = next(ln for ln in crit.splitlines() if ln.startswith("observed:"))

        fixed_case = solved = 0
        first_gen = ""
        for i in range(N):
            new = await repair(crit, entry, spec, sig, wrong, obs)
            nf = extract_entry_function(new, entry)
            if i == 0:
                first_gen = nf
            if not nf.strip():
                continue
            try:
                fn = _load(nf, entry)
            except Exception:  # noqa: BLE001
                continue
            # does it fix the CITED case X?
            import signal
            def _to(_s, _f):
                raise TimeoutError()
            signal.signal(signal.SIGALRM, _to)
            signal.alarm(5)
            try:
                cv = fn(*X)
            except Exception:  # noqa: BLE001
                cv = "<error>"
            finally:
                signal.alarm(0)
            if cv == want:
                fixed_case += 1
            if vc.first_failure(fn, all_cases) is None:
                solved += 1

        print(f"\n{'='*70}\n{qid} {entry}  (kind={kind})")
        print(f"  cited failing case: {obs[:100]}")
        print(f"  fixed_cited_case {fixed_case}/{N}   solved_all {solved}/{N}")
        print(f"  --- first repaired generation ---")
        for ln in first_gen.strip().splitlines()[:14]:
            print(f"    {ln}")


if __name__ == "__main__":
    asyncio.run(main())
