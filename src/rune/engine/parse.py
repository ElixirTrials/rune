"""Jinja2 template rendering and structured output parsing for engine actions."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader
from pydantic import BaseModel

from rune.engine.state import Action, Feedback, Subtask
from rune.sandbox.executor import extract_code

_env = Environment(loader=PackageLoader("rune", "templates"))


def render_template(template_name: str, **kwargs: Any) -> str:
    """Render a Jinja2 template from the rune/templates package directory.

    Args:
        template_name: Base name of the template (without .j2 extension).
        **kwargs: Context variables passed to the template.

    Returns:
        Rendered template string.
    """
    return _env.get_template(f"{template_name}.j2").render(**kwargs)


class SubtaskSchema(BaseModel):
    """Pydantic schema for a single subtask in decompose output.

    Attributes:
        name: Unique subtask identifier.
        description: What the subtask should implement.
        depends_on: Names of prerequisite subtasks.
    """

    name: str
    description: str
    depends_on: list[str] = []


class DecomposeResult(BaseModel):
    """Structured output from the decompose action.

    Attributes:
        subtasks: Ordered list of subtask schemas.
    """

    subtasks: list[SubtaskSchema]


class DiagnoseResult(BaseModel):
    """Structured output from the diagnose action.

    Attributes:
        fix_guidance: Actionable guidance for fixing the failing code.
    """

    fix_guidance: str


def parse_output(
    action: Action,
    raw: str,
    feedback: Feedback | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Parse raw model output into a partial RunState update dict.

    Args:
        action: The action that produced the raw output.
        raw: Raw text (or JSON) returned by the model.
        feedback: Sandbox result if the action executed code, otherwise None.
        state: Current RunState as a plain dict for reading existing values.

    Returns:
        Partial dict suitable for merging into RunState.
    """
    match action.name:
        case "decompose":
            result = DecomposeResult.model_validate_json(raw)
            return {
                "subtasks": [
                    Subtask(
                        name=s.name, description=s.description, depends_on=s.depends_on
                    )
                    for s in result.subtasks
                ]
            }
        case "plan":
            target = action.target_subtask
            return {"plans": {**state.get("plans", {}), target: raw}}
        case "code" | "code_retry":
            target = action.target_subtask
            passed = feedback is not None and feedback.exit_code == 0
            retries = dict(state.get("retries", {}))
            if action.name == "code_retry":
                retries[target] = retries.get(target, 0) + 1
            return {
                "code_results": {
                    **state.get("code_results", {}),
                    target: extract_code(raw),
                },
                "code_passed": {**state.get("code_passed", {}), target: passed},
                "retries": retries,
                "feedback": feedback,
            }
        case "integrate":
            passed = feedback is not None and feedback.exit_code == 0
            return {
                "integrated_code": extract_code(raw) if passed else "",
                "feedback": feedback,
                "diagnosis": None,
            }
        case "diagnose":
            diag = DiagnoseResult.model_validate_json(raw)
            return {"diagnosis": diag.fix_guidance}
    return {}
