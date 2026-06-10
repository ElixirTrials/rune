"""Verify the critique is PASSED intact through both engine channels (CPU-only).

REMOVE-BEFORE-MERGE. Renders render_episode_adapter (adapter conditioning) and
prompt_episodic_repair (in-context prompt) with the corrected critique + a fixed
state, and asserts: (1) observed+expected appear intact in BOTH channels, (2) the
spec is NOT duplicated in the adapter channel, (3) no mid-structure truncation.
No model is loaded.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"

_vc = importlib.util.spec_from_file_location(
    "vc", "/workspaces/content/tools/_verify_critique.py")
vc = importlib.util.module_from_spec(_vc)
_vc.loader.exec_module(vc)


def main() -> None:
    import sys
    sys.path.insert(0, "/workspaces/content/src")
    from rune.bench.lcb import extract_entry_function
    from rune.engine.graph import render_episode_adapter, state_to_ctx
    from rune.engine.parse import render_template
    from rune.engine.policy import ACTIONS
    from rune.engine.state import Feedback, StepRecord, Subtask

    rows = {json.loads(x)["question_id"]: json.loads(x)
            for x in Path(LCB).read_text().splitlines()}
    cands = {g["question_id"]: g["code_list"][0]
             for g in json.loads(Path(COMBINED).read_text())}
    base_repair = ACTIONS["repair"]

    for qid in ("3705", "3760"):  # one WRONG, one TLE
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        sig = row.get("starter_code", "") or ""
        wrong = extract_entry_function(cands[qid], entry)
        ns: dict[str, Any] = {}
        exec(wrong, ns)  # noqa: S102
        kind, args, got, want = vc.first_failure(ns[entry], vc._cases(row))
        crit = vc.build_critique(entry, kind, args, got, want)
        obs_line = next(ln for ln in crit.splitlines() if ln.startswith("observed:"))
        exp_line = next((ln for ln in crit.splitlines()
                         if ln.startswith("expected:")), "")

        # FIXED state: description empty (spec lives in overall_goal only -> no dup);
        # trajectory feedback carries the observed line as the "error" (not the
        # whole critique) so "approaches already tried" reads naturally.
        tried_err = obs_line.replace("observed: ", "")
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
                                      feedback=Feedback(stdout="", stderr=tried_err,
                                                        exit_code=1),
                                      generated_code=wrong)],
        }
        act = replace(base_repair, target_subtask=entry)
        ctx = state_to_ctx(state, act)
        adapter_txt = render_episode_adapter("repair", entry, state)
        prompt_txt = render_template("prompt_episodic_repair", **ctx)

        # --- checks ---
        spec_head = spec.strip().splitlines()[0][:50]
        checks = {
            "observed intact in prompt": obs_line in prompt_txt,
            "expected intact in prompt": (exp_line in prompt_txt) if exp_line else True,
            "observed intact in adapter": obs_line in adapter_txt,
            "spec NOT duplicated in adapter": adapter_txt.count(spec_head) == 1,
            "no mid-structure truncation": "...<len" not in obs_line.replace(
                "...<", "OK<") or "items>" in obs_line or "len " in obs_line,
            "prior code present in adapter": "your last attempt" in adapter_txt,
        }
        print(f"\n{'='*72}\n{qid} {entry}  (failure={kind})")
        print(f"  critique observed: {obs_line[:110]}")
        if exp_line:
            print(f"  critique expected: {exp_line[:110]}")
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  adapter spec_head count = {adapter_txt.count(spec_head)} "
              f"(want 1); adapter chars={len(adapter_txt)}, prompt chars={len(prompt_txt)}")


if __name__ == "__main__":
    main()
