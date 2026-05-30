"""Decompose parsing: drop pure-chore subtasks without emptying the plan."""

from __future__ import annotations

import json

from rune.engine.parse import parse_output
from rune.engine.policy import ACTIONS


def _decompose(subtasks: list[dict]) -> list:
    raw = json.dumps({"subtasks": subtasks})
    out = parse_output(ACTIONS["decompose"], raw, None, {})
    return out["subtasks"]


def test_drops_chore_subtasks_when_real_work_remains() -> None:
    result = _decompose(
        [
            {"name": "implement", "description": "core algorithm", "depends_on": []},
            {
                "name": "Write unit tests",
                "description": "test the function",
                "depends_on": ["implement"],
            },
            {
                "name": "Add documentation",
                "description": "write docstrings",
                "depends_on": ["implement"],
            },
        ]
    )
    assert [s.name for s in result] == ["implement"]


def test_chore_dep_references_are_dropped() -> None:
    # A surviving subtask must not depend on a removed chore subtask.
    result = _decompose(
        [
            {"name": "models", "description": "data structures", "depends_on": []},
            {
                "name": "Add type hints",
                "description": "annotate signatures",
                "depends_on": [],
            },
            {
                "name": "logic",
                "description": "core algorithm",
                "depends_on": ["models", "Add type hints"],
            },
        ]
    )
    names = {s.name for s in result}
    assert names == {"models", "logic"}
    logic = next(s for s in result if s.name == "logic")
    assert logic.depends_on == ["models"]


def test_keeps_all_when_every_subtask_is_chore() -> None:
    # Never empty the plan — degrade to keeping everything.
    result = _decompose(
        [
            {
                "name": "Add documentation",
                "description": "docstrings",
                "depends_on": [],
            },
            {"name": "Write unit tests", "description": "tests", "depends_on": []},
        ]
    )
    assert len(result) == 2
