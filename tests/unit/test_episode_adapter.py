"""Episodic adapter conditioning: the right context per step (not the full spec)."""

from __future__ import annotations

from rune.engine.graph import render_episode_adapter
from rune.engine.state import Feedback, Subtask


def _state() -> dict:
    return {
        "task": '"""Implement calculate(expr) -> int. >>> assert calculate("2+3")==5"""',
        "entry_point": "calculate",
        "overall_goal": "Evaluate an arithmetic expression string to an int.",
        "subtasks": [
            Subtask(
                "tokenize",
                "Split the expression into tokens",
                [],
                "assert tokenize('2+3')==['2','+','3']",
                "calculate",
            ),
            Subtask(
                "evaluate",
                "Evaluate the token list",
                ["tokenize"],
                "assert evaluate(['2','+','3'])==5",
                "calculate",
            ),
        ],
        "code_results": {"tokenize": "def tokenize(s): return list(s)"},
        "feedback": {
            "tokenize": Feedback(
                stdout="", stderr="AssertionError: bad split", exit_code=1
            )
        },
        "diagnosis": {},
        "integration_feedback": None,
    }


class TestEpisodeAdapter:
    def test_code_step_is_focused_on_the_subgoal(self) -> None:
        adp = render_episode_adapter("code", "tokenize", _state())
        # focused: the current sub-goal + acceptance, the overall goal, the local state
        assert "tokenize" in adp
        assert "Split the expression into tokens" in adp
        assert "assert tokenize('2+3')" in adp  # the sub-goal's acceptance check
        assert "Evaluate an arithmetic expression" in adp  # condensed overall goal
        # NOT the full original spec / the OTHER subtask
        assert ">>> assert calculate" not in adp
        assert "Evaluate the token list" not in adp

    def test_code_step_carries_local_code_and_error(self) -> None:
        adp = render_episode_adapter("code", "tokenize", _state())
        assert "def tokenize(s): return list(s)" in adp  # ## Current Code
        assert "AssertionError: bad split" in adp  # ## Review Feedback

    def test_integration_step_carries_all_subtasks(self) -> None:
        st = _state()
        st["code_results"] = {
            "tokenize": "def tokenize(s): ...",
            "evaluate": "def evaluate(t): ...",
        }
        adp = render_episode_adapter("integrate", None, st)
        assert "calculate" in adp  # the entry_point to integrate into
        assert "def tokenize(s): ..." in adp  # ALL subtasks' code
        assert "def evaluate(t): ..." in adp

    def test_decompose_step_sees_the_full_spec(self) -> None:
        adp = render_episode_adapter("decompose", None, _state())
        assert ">>> assert calculate" in adp  # decompose needs the full task
