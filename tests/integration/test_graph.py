from __future__ import annotations

from unittest.mock import MagicMock

from rune.engine.graph import create_engine, should_continue
from rune.engine.state import RunState, Subtask


def _initial_state(task: str = "add two numbers", budget: int = 10) -> RunState:
    return {
        "task": task,
        "subtasks": [],
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {},
        "diagnosis": {},
        "actions": [MagicMock()],  # non-empty so first step runs
        "trajectory": [],
        "step": 0,
        "budget_remaining": budget,
    }


class TestShouldContinue:
    def test_solved_single_subtask_returns_done(self) -> None:
        state = _initial_state()
        state["subtasks"] = [Subtask("_main", "do main", [])]
        state["plans"] = {"_main": "plan"}
        state["code_passed"] = {"_main": True}
        state["code_results"] = {"_main": "def f(): pass"}
        assert should_continue(state) == "done"

    def test_budget_zero_returns_done(self) -> None:
        state = _initial_state(budget=0)
        assert should_continue(state) == "done"

    def test_has_actions_and_budget_returns_continue(self) -> None:
        state = _initial_state()
        assert should_continue(state) == "continue"


class TestCreateEngine:
    def test_engine_compiles(self) -> None:
        engine = create_engine()
        assert engine is not None
