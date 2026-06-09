"""Golden tests from q3753 B4 repair-signal failures (issue #52).

Each test encodes a concrete regression from /tmp/goal3/ab/b4_v2_sessions/3753
or the offline eval harness — the signal the full stack must preserve or improve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rune.engine.graph import render_episode_adapter, state_to_ctx
from rune.engine.parse import (
    Action,
    DiagnoseResult,
    parse_output,
    render_template,
)
from rune.engine.policy import select_action
from rune.engine.repair_brief import build_repair_brief, merge_guidance_with_brief
from rune.engine.state import Feedback, StepRecord, Subtask, make_initial_state

B4_SESSION = Path("/tmp/goal3/ab/b4_v2_sessions/3753/session.jsonl")
SIG_3753 = "class Solution:\n    def maxDifference(self, s: str) -> int:\n        "
PLAN_3753 = (
    "Count the frequency of each character in the string. Identify all characters "
    "with odd frequencies and all with even frequencies. Find the maximum difference "
    "between any odd-frequency character's count and any even-frequency character's count."
)
ODD_EVEN_CODE = """class Solution:
    def maxDifference(self, s: str) -> int:
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        odd_freq = []
        even_freq = []
        for char, count in freq.items():
            if count % 2 == 0:
                even_freq.append(count)
            else:
                odd_freq.append(count)
        max_diff = 0
        for odd_count in odd_freq:
            for even_count in even_freq:
                max_diff = max(max_diff, odd_count - even_count)
        return max_diff"""


def _b4_stderr(step: int) -> str:
    if not B4_SESSION.exists():
        raise pytest.skip.Exception(f"missing {B4_SESSION}")
    for line in B4_SESSION.read_text().splitlines():
        rec = json.loads(line)
        if rec["step"] == step:
            return str((rec.get("feedback") or {}).get("stderr", ""))
    raise pytest.skip.Exception(f"step {step} missing in B4 session")


def _3753_state(*, with_passing_code: bool = False) -> dict:
    state = make_initial_state(
        "task", 12, "maxDifference", SIG_3753, "assert maxDifference('aaaaabbc') == 3"
    )
    state.update(
        {
            "overall_goal": "Find max odd-even frequency difference in a string.",
            "subtasks": [
                Subtask(
                    name="maxDifference",
                    description=PLAN_3753,
                    depends_on=[],
                    acceptance_check="assert maxDifference('aaaaabbc') == 3",
                    builds="maxDifference",
                )
            ],
            "plans": {"maxDifference": PLAN_3753},
            "code_results": {"maxDifference": ODD_EVEN_CODE},
            "code_passed": {"maxDifference": with_passing_code},
            "code_solved": {"maxDifference": with_passing_code},
            "best_code": {"maxDifference": ODD_EVEN_CODE},
            "retries": {"maxDifference": 1},
        }
    )
    return state


class TestAssertionBriefEnrichment:
    def test_assertion_brief_carries_failing_case_task_agnostic(self) -> None:
        # The brief must carry the concrete observed-vs-expected failing case and
        # must NOT inject any task-specific solution (overfitting; misfires on
        # other tasks). See test_repair_brief.test_assertion_brief_is_task_agnostic.
        stderr = _b4_stderr(4)
        brief = build_repair_brief(
            stderr, entry_point="maxDifference", signature=SIG_3753
        )
        assert brief is not None
        assert brief.failure_class == "assertion"
        assert brief.observed.strip()
        assert brief.expected.strip()
        blob = brief.format_block().lower()
        for leaked in ("odd-frequency", "not max(all_freq)", "anti-diagonal"):
            assert leaked not in blob


class TestMergeGuidanceHygiene:
    def test_b4_step5_llm_guidance_appended_as_advisory(self) -> None:
        # Task-agnostic merge: the deterministic brief leads and the model's
        # how-to-fix is appended as advisory (never suppressed by task keywords).
        brief = build_repair_brief(
            _b4_stderr(4), entry_point="maxDifference", signature=SIG_3753
        )
        assert brief is not None
        llm = (
            "Identify the highest and lowest frequencies in the character count "
            "map and return their absolute difference."
        )
        merged = merge_guidance_with_brief(brief.format_block(), llm)
        assert "failure_class: assertion" in merged
        assert "how_to_fix" in merged

    def test_additive_guidance_kept_when_non_contradictory(self) -> None:
        brief = build_repair_brief(
            "Task requirements failed — fix exactly:\n- signature: expected def maxDifference(s)",
            entry_point="maxDifference",
            signature=SIG_3753,
        )
        assert brief is not None
        llm = "Remove self and define def maxDifference(s):"
        merged = merge_guidance_with_brief(brief.format_block(), llm)
        assert "how_to_fix" in merged or "Remove self" in merged
        assert "failure_class: signature" in merged


class TestPreserveLogicDirective:
    def test_signature_brief_preserves_algorithm(self) -> None:
        stderr = _b4_stderr(2)
        brief = build_repair_brief(
            stderr, entry_point="maxDifference", signature=SIG_3753
        )
        assert brief is not None
        assert brief.failure_class == "signature"
        assert "preserve" in brief.fix_directive.lower()

    def test_episodic_repair_prompt_has_preserve_logic_for_signature(self) -> None:
        brief = build_repair_brief(
            _b4_stderr(2), entry_point="maxDifference", signature=SIG_3753
        )
        assert brief is not None
        prompt = render_template(
            "prompt_episodic_repair",
            subtask_name="maxDifference",
            bare_signature="def maxDifference(s: str) -> int:",
            repair_brief=brief.format_block(),
            fix_guidance="Remove self parameter",
            preserve_logic=True,
        )
        assert "preserve" in prompt.lower()
        assert "algorithm" in prompt.lower() or "logic" in prompt.lower()


class TestEpisodicRepairContext:
    def test_repair_prompt_includes_fix_guidance(self) -> None:
        prompt = render_template(
            "prompt_episodic_repair",
            subtask_name="maxDifference",
            bare_signature="def maxDifference(s: str) -> int:",
            repair_brief="failure_class: assertion",
            fix_guidance="Use odd vs even frequency counts, not max-min of all freqs",
        )
        assert "odd vs even" in prompt

    def test_repair_prompt_includes_repair_history(self) -> None:
        prompt = render_template(
            "prompt_episodic_repair",
            subtask_name="maxDifference",
            bare_signature="def maxDifference(s: str) -> int:",
            repair_history=[
                "AssertionError: maxDifference(*['aaaaabbc']) -> 4, want 3",
                "AssertionError: same failure again",
            ],
        )
        low = prompt.lower()
        assert "do not retry" in low or "prior" in low or "already tried" in low
        assert "aaaaabbc" in prompt

    def test_adapter_includes_tried_and_failed(self) -> None:
        state = _3753_state()
        fail_fb = Feedback(
            stdout="",
            stderr="AssertionError: maxDifference(*['aaaaabbc']) -> 4, want 3",
            exit_code=1,
        )
        state["diagnosis"] = {
            "maxDifference": "failure_class: assertion\nobserved: 4\nexpected: 3"
        }
        state["feedback"] = {"maxDifference": fail_fb}
        state["trajectory"] = [
            StepRecord(
                step=4,
                action_name="repair",
                target_subtask="maxDifference",
                adapter_id="a",
                feedback=fail_fb,
                generated_code="def maxDifference(s): return max(freq)-min(freq)",
            ),
            StepRecord(
                step=6,
                action_name="repair",
                target_subtask="maxDifference",
                adapter_id="a",
                feedback=fail_fb,
                generated_code="def maxDifference(s): return max(freq)-min(freq)",
            ),
        ]
        adp = render_episode_adapter("repair", "maxDifference", state)
        low = adp.lower()
        assert "tried" in low or "already" in low or "do not retry" in low


class TestCodeSolvedLatch:
    def test_diagnose_does_not_reopen_solved_subtask(self) -> None:
        state = _3753_state(with_passing_code=True)
        action = Action(
            "diagnose", "diagnose", "prompt_diagnose", "", DiagnoseResult, False, None
        )
        raw = json.dumps(
            {
                "entries": [
                    {
                        "subtask_name": "maxDifference",
                        "error_type": "logic",
                        "location": "integration",
                        "fix_guidance": "reopen and fix",
                    }
                ]
            }
        )
        out = parse_output(action, raw, None, state)
        assert out.get("code_passed", {}).get("maxDifference") is True

    def test_policy_skips_solved_subtask(self) -> None:
        state = _3753_state(with_passing_code=True)
        state["diagnosis"] = {"maxDifference": "stale guidance"}
        actions = select_action(state)
        assert actions == []


class TestStateToCtxPreserveLogic:
    def test_signature_failure_sets_preserve_logic_flag(self) -> None:
        state = _3753_state()
        brief = build_repair_brief(
            _b4_stderr(2), entry_point="maxDifference", signature=SIG_3753
        )
        assert brief is not None
        state["repair_briefs"] = {"maxDifference": brief.format_block()}
        action = Action(
            "repair",
            "code_repair",
            "prompt_episodic_repair",
            "",
            None,
            True,
            "maxDifference",
        )
        ctx = state_to_ctx(state, action)
        assert ctx.get("preserve_logic") is True
