from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader
from pydantic import BaseModel

from rune.engine.state import Action, Feedback, Subtask
from rune.sandbox.executor import extract_code

_env = Environment(loader=PackageLoader("rune", "templates"))


def render_template(template_name: str, **kwargs: Any) -> str:
    return _env.get_template(f"{template_name}.j2").render(**kwargs)


class SubtaskSchema(BaseModel):
    name: str
    description: str
    depends_on: list[str] = []


class DecomposeResult(BaseModel):
    subtasks: list[SubtaskSchema]


class DiagnoseResult(BaseModel):
    fix_guidance: str


def parse_output(
    action: Action,
    raw: str,
    feedback: Feedback | None,
    state: dict[str, Any],
) -> dict[str, Any]:
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
