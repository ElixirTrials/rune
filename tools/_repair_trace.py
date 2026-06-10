"""Forensic trace: EXACTLY how the oracle passes info to the model, and EXACTLY
how the model's output changes in response.

REMOVE-BEFORE-MERGE. For a recoverer (3705) and a never-solver (3717), dumps:
  1. the perfect critique (the corrective signal),
  2. CHANNEL 1 -- adapter conditioning text (baked into LoRA weights), full,
  3. CHANNEL 2 -- in-context repair prompt, full,
  4. the ORIGINAL code,
  5. the RAW model generation, full (untruncated),
  6. a unified DIFF original -> new code,
  7. verdict: did it fix the CITED case? solve ALL? and if not, the FIRST case the
     NEW code still gets wrong (got vs expected) -- i.e. how the change fell short.
"""

from __future__ import annotations

import asyncio
import difflib
import importlib.util
import json
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
TASKS = ["3705", "3717"]

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


def _run_on(code: str, entry: str, X: list):
    fn = _load(code, entry)

    def _to(_s, _f):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _to)
    signal.alarm(5)
    try:
        return repr(fn(*X))
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"
    finally:
        signal.alarm(0)


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

    for qid in TASKS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        sig = row.get("starter_code", "") or ""
        all_cases = vc._cases(row)
        wrong = extract_entry_function(cands[qid], entry)
        kind, X, got, want = vc.first_failure(_load(wrong, entry), all_cases)
        crit = vc.build_critique(entry, kind, X, got, want)
        obs = next(ln for ln in crit.splitlines() if ln.startswith("observed:"))
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
        adapter_txt = render_episode_adapter("repair", entry, state)
        prompt = render_template("prompt_episodic_repair", **ctx)
        sc = _effective_scaling("escalate", act, state["code_results"], 1.0)
        apply_episodic_adapter(model, traj := adapter_txt, scaling=sc)
        gen = await model.generate(prompt=prompt, system_prompt=act.system_prompt,
                                   max_tokens=2048, temperature=0.3, thinking_budget=0)
        raw = gen.text
        new = extract_entry_function(extract_code_block(raw) or "", entry)

        B = "#" * 80
        print(f"\n{B}\n# {qid} {entry}   (adapter scaling={sc})\n{B}")
        print(f"\n========== (1) PERFECT CRITIQUE — the corrective signal ==========\n{crit}")
        print(f"\n========== (2) CHANNEL 1: ADAPTER CONDITIONING (LoRA weights) "
              f"[{len(adapter_txt)} chars] ==========\n{adapter_txt}")
        print(f"\n========== (3) CHANNEL 2: IN-CONTEXT REPAIR PROMPT "
              f"(system={act.system_prompt!r}) [{len(prompt)} chars] ==========\n{prompt}")
        print(f"\n========== (4) ORIGINAL (wrong) CODE ==========\n{wrong}")
        print(f"\n========== (5) RAW MODEL GENERATION [{len(raw)} chars] ==========\n{raw}")
        print(f"\n========== (6) DIFF  original -> new ==========")
        diff = difflib.unified_diff(wrong.splitlines(), new.splitlines(),
                                    "original", "repaired", lineterm="")
        print("\n".join(diff) or "  (no change)")
        print(f"\n========== (7) VERDICT ==========")
        cited_now = _run_on(new, entry, X) if new.strip() else "<empty>"
        print(f"  cited case X: {vc._summarize(X)[:100]}")
        print(f"  expected={want!r}  original_gave={got!r}  new_gives={cited_now}")
        print(f"  fixed_cited_case = {cited_now == repr(want)}")
        if new.strip():
            still = vc.first_failure(_load(new, entry), all_cases)
            if still is None:
                print("  solved_all = True")
            else:
                k2, X2, g2, w2 = still
                print(f"  solved_all = False; NEW code's FIRST failure: {k2} on "
                      f"{vc._summarize(X2)[:80]} -> got {g2!r}, expected {w2!r}")


if __name__ == "__main__":
    asyncio.run(main())
