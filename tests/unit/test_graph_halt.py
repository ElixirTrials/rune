"""Engine must halt when select_action has no further work (e.g. subtask solved)."""

from __future__ import annotations

from rune.engine.graph import should_continue
from rune.engine.policy import select_action
from rune.engine.state import Action, Subtask


def _single_passed_state() -> dict:
    return {
        "task": "test",
        "subtasks": [Subtask("maxDifference", "solve", [])],
        "plans": {"maxDifference": "plan"},
        "code_results": {"maxDifference": "def maxDifference(s): return 1"},
        "code_passed": {"maxDifference": True},
        "retries": {"maxDifference": 1},
        "integrated_code": "def maxDifference(s): return 1",
        "current_adapter": None,
        "feedback": {},
        "integration_feedback": None,
        "diagnosis": {},
        "actions": [
            Action(
                "repair",
                "code_repair",
                "prompt_code_repair",
                "",
                None,
                True,
                "maxDifference",
            )
        ],
        "trajectory": [],
        "step": 4,
        "budget_remaining": 8,
    }


class TestShouldContinueHalt:
    def test_halts_when_single_subtask_passed_even_if_stale_actions(self) -> None:
        """Regression: state.actions held the repair that just passed, not next work."""
        state = _single_passed_state()
        assert select_action(state) == []
        assert should_continue(state) == "done"

    def test_continues_while_failing_subtask_has_work(self) -> None:
        state = _single_passed_state()
        state["code_passed"] = {"maxDifference": False}
        state["integrated_code"] = ""
        state["diagnosis"] = {"maxDifference": "fix it"}
        assert select_action(state)[0].name == "repair"
        assert should_continue(state) == "continue"
