#!/usr/bin/env python3
"""Offline per-fix certainty metrics for repair-signal changes (no GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rune.engine.parse import render_template
from rune.engine.plan_gate import validate_plan
from rune.engine.policy import select_action
from rune.engine.repair_brief import build_repair_brief, merge_guidance_with_brief
from rune.engine.state import Subtask, make_initial_state

SESSIONS = Path("/tmp/goal3/rerun_failures2/sessions")
SIGNATURES: dict[str, str] = {
    "3754": "class Solution:\n    def maxDistance(self, s: str, k: int) -> int:\n        ",
    "3748": "class Solution:\n    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:\n        ",
    "3777": "class Solution:\n    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:\n        ",
    "3799": "class Solution:\n    def totalNumbers(self, digits: List[int]) -> int:\n        ",
    "3801": "class Solution:\n    def beautifulNumbers(self, l: int, r: int) -> int:\n        ",
    "3753": "class Solution:\n    def maxDifference(self, s: str) -> int:\n        ",
}
ENTRY: dict[str, str] = {
    "3754": "maxDistance",
    "3748": "sortMatrix",
    "3777": "maxProduct",
    "3799": "totalNumbers",
    "3801": "beautifulNumbers",
    "3753": "maxDifference",
}
TASK_SNIPPETS: dict[str, str] = {
    "3777": "find a non-empty subsequence of nums",
    "3799": "You are given an array of digits",
}


def _plan(qid: str) -> str:
    for line in (SESSIONS / qid / "session.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["action"] == "plan":
            out = rec["output"]
            return json.loads(out)["plan"] if out.startswith("{") else out
    raise KeyError(qid)


def _stderr(qid: str, step: int) -> str:
    for line in (SESSIONS / qid / "session.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["step"] == step:
            fb = rec.get("feedback") or {}
            return str(fb.get("stderr", ""))
    raise KeyError((qid, step))


def eval_repair_brief() -> dict[str, Any]:
    cases = [
        ("3753", 2, "signature", False),
        ("3777", 2, "complexity", True),
        ("3801", 2, "signature", False),
        ("3754", 4, "arity", False),
        ("3799", 2, "import", False),
        ("3748", 2, "assertion", False),
    ]
    ok = 0
    rows: list[dict[str, Any]] = []
    for qid, step, want_class, want_replan in cases:
        brief = build_repair_brief(
            _stderr(qid, step),
            entry_point=ENTRY[qid],
            signature=SIGNATURES[qid],
        )
        hit = (
            brief is not None
            and brief.failure_class == want_class
            and brief.replan_recommended is want_replan
        )
        ok += int(hit)
        rows.append(
            {
                "qid": qid,
                "step": step,
                "want": want_class,
                "got": brief.failure_class if brief else None,
                "ok": hit,
            }
        )
    return {"fix": "repair_brief", "accuracy": ok / len(cases), "rows": rows}


def eval_plan_gate() -> dict[str, Any]:
    bad = ["3754", "3777", "3799", "3801"]
    good = ["3753", "3748"]
    caught = 0
    fp = 0
    for qid in bad:
        gate = validate_plan(
            _plan(qid),
            entry_point=ENTRY[qid],
            signature=SIGNATURES[qid],
            task_spec=TASK_SNIPPETS.get(qid, ""),
        )
        if not gate.ok:
            caught += 1
    for qid in good:
        gate = validate_plan(
            _plan(qid),
            entry_point=ENTRY[qid],
            signature=SIGNATURES[qid],
            task_spec=TASK_SNIPPETS.get(qid, ""),
        )
        if not gate.ok:
            fp += 1
    return {
        "fix": "plan_gate",
        "bad_plan_recall": caught / len(bad),
        "good_plan_fp_rate": fp / len(good),
        "caught": caught,
        "false_positives": fp,
    }


def eval_replan_routing() -> dict[str, Any]:
    state = make_initial_state("t", 12, "maxProduct", SIGNATURES["3777"])
    state["subtasks"] = [
        Subtask(
            name="maxProduct",
            description="d",
            acceptance_check="assert maxProduct([[1]], 1, 1) == 1",
            builds="maxProduct",
            depends_on=[],
        )
    ]
    state["plans"] = {"maxProduct": "bad plan"}
    state["code_passed"] = {"maxProduct": False}
    state["code_results"] = {"maxProduct": "def maxProduct(nums, k, limit): pass"}
    state["retries"] = {"maxProduct": 1}
    state["replan_targets"] = {"maxProduct": True}
    state["plans"] = {}
    actions = select_action(state)
    routed_plan = actions[0].name == "plan" if actions else False
    return {"fix": "replan_routing", "3777_routes_to_plan": routed_plan}


def eval_repair_prompt_signal() -> dict[str, Any]:
    brief = build_repair_brief(
        _stderr("3748", 2),
        entry_point="sortMatrix",
        signature=SIGNATURES["3748"],
    )
    assert brief is not None
    prompt = render_template(
        "prompt_episodic_repair",
        subtask_name="sortMatrix",
        bare_signature="def sortMatrix(grid):",
        repair_brief=brief.format_block(),
    )
    has_invariant = "anti-diagonal" in prompt.lower()
    has_sig = "def sortMatrix(grid)" in prompt
    return {
        "fix": "repair_prompt",
        "3748_invariant_in_prompt": has_invariant,
        "signature_anchor_present": has_sig,
    }


def eval_3753_assertion_enrichment() -> dict[str, Any]:
    """B4 step-4: assertion brief must name odd/even parity, not generic logic."""
    plan = _plan("3753")
    brief = build_repair_brief(
        _stderr("3753", 4),
        entry_point=ENTRY["3753"],
        signature=SIGNATURES["3753"],
        plan=plan,
    )
    inv = (brief.violated_invariant if brief else "").lower()
    ok = brief is not None and "odd" in inv and "even" in inv
    return {
        "fix": "assertion_enrichment_3753",
        "odd_even_invariant": ok,
        "invariant": brief.violated_invariant if brief else None,
    }


def eval_merge_hygiene_3753() -> dict[str, Any]:
    plan = _plan("3753")
    brief = build_repair_brief(
        _stderr("3753", 4),
        entry_point=ENTRY["3753"],
        signature=SIGNATURES["3753"],
        plan=plan,
    )
    assert brief is not None
    llm = (
        "identifying the highest and lowest frequencies in the character count map"
    )
    merged = merge_guidance_with_brief(brief.format_block(), llm)
    filtered = "highest and lowest" not in merged
    return {"fix": "merge_hygiene_3753", "misleading_guidance_filtered": filtered}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    results = [
        eval_repair_brief(),
        eval_plan_gate(),
        eval_replan_routing(),
        eval_repair_prompt_signal(),
        eval_3753_assertion_enrichment(),
        eval_merge_hygiene_3753(),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
