"""Dump EXACTLY what the perfect-oracle forced repair conveyed to the model.

REMOVE-BEFORE-MERGE. For a few false-pass tasks, reconstruct the real engine
repair step (perfect critique = true hidden failing case) and print BOTH signal
channels verbatim:
  1. adapter conditioning text (render_episode_adapter) -- baked into LoRA weights
  2. in-context prompt (prompt_episodic_repair) -- what the model reads
  3. the model's raw generation.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import json
import pickle
import zlib
from dataclasses import replace
from pathlib import Path

LCB = "/tmp/lcb/test6.jsonl"
COMBINED = "/tmp/goal3/overnight/lcb_postfix_combined.json"
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
TASKS = ["3705", "3760"]


def _decode_private(s: str) -> list:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode()))))


def _all_cases(row: dict) -> list:
    raw = json.loads(row["public_test_cases"]) + _decode_private(
        row["private_test_cases"]
    )
    out = []
    for t in raw:
        try:
            a = [ast.literal_eval(x) for x in t["input"].split("\n") if x.strip()]
            out.append((a, ast.literal_eval(t["output"])))
        except (ValueError, SyntaxError):
            continue
    return out


_GRADE = """\
from typing import *
import collections, math, heapq, bisect, itertools, functools, re, signal
from collections import defaultdict, deque, Counter, OrderedDict
import base64
exec(compile(base64.b64decode({code_b64!r}).decode(), '<c>', 'exec'), globals())
cand = globals()[{entry!r}]
cases = {cases!r}
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
first = None
for i, (args, exp) in enumerate(cases):
    signal.alarm(6)
    try:
        got = cand(*args); signal.alarm(0)
        if got != exp:
            first = (i, 'WRONG', repr(args)[:200], repr(got)[:120], repr(exp)[:120]); break
    except TimeoutError:
        first = (i, 'TLE', repr(args)[:120], '', ''); break
    except Exception as e:
        signal.alarm(0)
        first = (i, 'CRASH:' + type(e).__name__, repr(args)[:200], '', repr(exp)[:120]); break
print('SOLVED' if first is None else 'FAIL', repr(first) if first else '')
"""  # noqa: E501


async def main() -> None:
    from rune.bench.lcb import extract_entry_function  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.graph import (  # noqa: PLC0415
        _effective_scaling,
        render_episode_adapter,
        state_to_ctx,
    )
    from rune.engine.oracle import with_probe_imports  # noqa: PLC0415
    from rune.engine.parse import extract_code_block, render_template  # noqa: PLC0415
    from rune.engine.policy import ACTIONS  # noqa: PLC0415
    from rune.engine.state import Feedback, StepRecord, Subtask  # noqa: PLC0415
    from rune.model.adapter import apply_episodic_adapter  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415

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

    def first_hidden_fail(code, entry, cases):
        nc = extract_entry_function(code, entry)
        script = _GRADE.format(
            code_b64=base64.b64encode(nc.encode()).decode(), entry=entry, cases=cases
        )
        out = (
            run_in_sandbox(with_probe_imports(script), timeout=90).stdout or ""
        ).strip()
        if out.startswith("SOLVED"):
            return None
        return eval(out.split(" ", 1)[1])  # noqa: S307

    for qid in TASKS:
        row = rows[qid]
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        entry = meta.get("func_name") or ""
        spec = row.get("question_content", "")[:3500]
        sig = row.get("starter_code", "") or ""
        cases = _all_cases(row)
        wrong = extract_entry_function(cands.get(qid, ""), entry)
        fail = first_hidden_fail(wrong, entry, cases)
        _i, kind, inp, got, want = fail
        crit = (
            f"failure_class: assertion\nobserved: {entry}({inp}) -> {got}\n"
            f"expected: {want}\nfix_directive: fix the algorithm so observed "
            f"output matches expected."
        )
        state = {
            "entry_point": entry,
            "signature": sig,
            "task": spec,
            "public_checks": "",
            "overall_goal": spec,
            "subtasks": [
                Subtask(
                    name=entry,
                    description=spec[:600],
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
                    feedback=Feedback(stdout="", stderr=crit, exit_code=1),
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
        new = extract_entry_function(extract_code_block(gen.text) or "", entry)

        bar = "#" * 78
        print(f"\n{bar}\n# {qid}  {entry}   (adapter scaling={scaling})\n{bar}")
        print(f"\n----- PERFECT CRITIQUE (the corrective signal) -----\n{crit}")
        print(
            f"\n----- CHANNEL 1: ADAPTER CONDITIONING (baked into LoRA weights) -----\n{traj}"  # noqa: E501
        )
        print(
            f"\n----- CHANNEL 2: IN-CONTEXT REPAIR PROMPT (system='{act.system_prompt}') -----\n{prompt}"  # noqa: E501
        )
        print(
            f"\n----- MODEL GENERATION (raw, len={len(gen.text)}) -----\n{gen.text[:2400]}"  # noqa: E501
        )
        print(
            f"\n----- EXTRACTED FN  (changed={new.strip() != wrong.strip()}) -----\n{new[:1200]}"  # noqa: E501
        )


if __name__ == "__main__":
    asyncio.run(main())
