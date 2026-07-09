"""Render-mechanics tests for every engine template."""

from __future__ import annotations

import jinja2.meta
import pytest

from rune.engine.parse import _env, render_template
from rune.engine.state import Subtask

_SUBTASK = Subtask(name="_main", description="do it", depends_on=[])


def _ctx() -> dict[str, object]:
    """Superset of keys produced by state_to_ctx, for branch coverage."""
    return {
        "project": "build a thing",
        "task_description": "build a thing",
        "project_label": "build a thing",
        "subtask_count": 1,
        "entry_point": "do_it",
        "subtask": _SUBTASK,
        "subtask_name": "_main",
        "subtask_index": 1,
        "total_subtasks": 1,
        "plan": "the plan",
        "target_subtask": "_main",
        "existing_code": "print(1)",
        "error_summary": "boom",
        "repair_brief": "",
        "repair_context": False,
        "concise_code": False,
        "last_failure": "",
        "fix_guidance": "fix it",
        "repair_history": ["err"],
        "code_trajectory": [
            {"step": 0, "action": "code", "code": "x=1", "error": "", "passed": True}
        ],
        "integration_doc": "- _main: do it",
        "code_outputs": {"_main": "print(1)"},
        "integration_error": "",
        "accumulated_code": "print(1)",
    }


_TEMPLATES = [
    "decompose",
    "prompt_decompose_concise",
    "plan",
    "prompt_plan",
    "code",
    "prompt_code",
    "code_repair",
    "prompt_code_repair",
    "integrate",
    "prompt_integrate",
    "diagnose",
    "prompt_diagnose",
    "code_continue",
    "prompt_code_continue",
]


@pytest.mark.parametrize("name", _TEMPLATES)
def test_renders_without_undefined(name: str) -> None:
    render_template(name, **_ctx())


@pytest.mark.parametrize("name", _TEMPLATES)
def test_declared_vars_are_supplied(name: str) -> None:
    source = _env.loader.get_source(_env, f"{name}.j2")[0]
    declared = jinja2.meta.find_undeclared_variables(_env.parse(source))
    missing = declared - set(_ctx())
    assert not missing, f"{name}.j2 needs unsupplied vars: {missing}"


@pytest.mark.parametrize("name", ["diagnose", "prompt_diagnose", "code_continue"])
def test_renders_with_no_target(name: str) -> None:
    ctx = _ctx()
    ctx.update({"subtask": None, "target_subtask": None})
    render_template(name, **ctx)


def test_prompt_code_names_entry_point_when_present() -> None:
    ctx = _ctx()
    ctx["entry_point"] = "add_lists"
    out = render_template("prompt_code", **ctx)
    assert "add_lists" in out
    # neutral on tests: we neither force "write tests" nor forbid them
    assert "tests FIRST" not in out
    assert "no tests" not in out.lower()


def test_prompt_code_omits_function_line_without_entry_point() -> None:
    ctx = _ctx()
    ctx["entry_point"] = ""
    out = render_template("prompt_code", **ctx)
    assert "Implement the function" not in out
