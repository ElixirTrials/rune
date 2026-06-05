"""Robust structured parsing for decompose/plan/diagnose + episodic decompose schema.

The JSON-structured decompose/plan/diagnose steps truncated on long hard-task output
-> pydantic validation failed -> re-plan/re-decompose loop -> empty code (3/4 empty on
the LiveCodeBench probe). These tests pin: (1) json_repair recovers truncated/garbled
JSON; (2) the episodic decompose schema (overall_goal, acceptance_check, builds);
(3) the <=3 cap; (4) graceful degrade to a single whole-task subtask instead of an
empty re-decompose loop.
"""

from __future__ import annotations

from rune.engine.parse import (
    DecomposeResult,
    SubtaskSchema,
    _loads_structured,
    parse_output,
)
from rune.engine.state import Action


def _decompose_action() -> Action:
    return Action(
        "decompose", "decompose", "prompt_decompose_concise", "", DecomposeResult,
        False, None,
    )


def _state(task: str = "Write add(a,b) returning a+b.") -> dict:
    return {"subtasks": [], "task": task, "entry_point": "add"}


class TestLoadsStructured:
    def test_valid_json(self) -> None:
        raw = '{"subtasks": [{"name": "a", "description": "do a", "depends_on": []}]}'
        res = _loads_structured(raw, DecomposeResult)
        assert res is not None
        assert res.subtasks[0].name == "a"

    def test_truncated_json_recovered(self) -> None:
        # truncated mid-object (no closing braces) -> json_repair recovers it
        raw = '{"subtasks": [{"name": "core", "description": "the whole thing"'
        res = _loads_structured(raw, DecomposeResult)
        assert res is not None
        assert res.subtasks and res.subtasks[0].name == "core"

    def test_unrecoverable_returns_none(self) -> None:
        assert _loads_structured("total garbage no json here", DecomposeResult) is None


class TestEpisodicSchema:
    def test_subtask_new_fields_default(self) -> None:
        s = SubtaskSchema(name="a", description="d")
        assert s.acceptance_check == ""
        assert s.builds == ""

    def test_subtask_accepts_episodic_fields(self) -> None:
        s = SubtaskSchema(
            name="tok", description="tokenizer",
            acceptance_check="assert tokenize('1+2') == ['1','+','2']", builds="calculate",
        )
        assert s.acceptance_check.startswith("assert")
        assert s.builds == "calculate"

    def test_decompose_overall_goal_default(self) -> None:
        d = DecomposeResult(subtasks=[SubtaskSchema(name="a", description="d")])
        assert d.overall_goal == ""


class TestDecomposeBoundAndDegrade:
    def test_caps_to_three_subtasks(self) -> None:
        raw = (
            '{"subtasks": ['
            '{"name":"a","description":"da"},{"name":"b","description":"db"},'
            '{"name":"c","description":"dc"},{"name":"d","description":"dd"},'
            '{"name":"e","description":"de"}]}'
        )
        updates = parse_output(_decompose_action(), raw, None, _state())
        assert len(updates["subtasks"]) <= 3

    def test_garbled_degrades_to_single_whole_task(self) -> None:
        # unparseable decompose must NOT return {} (which re-decompose-loops);
        # it degrades to ONE subtask = the whole task.
        updates = parse_output(_decompose_action(), "??? not json ???", None, _state())
        assert len(updates.get("subtasks", [])) == 1
