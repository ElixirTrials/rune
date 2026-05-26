"""Jinja2 template rendering and structured output parsing for engine actions."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from rune.engine.state import Action, Feedback, Subtask
from rune.sandbox.executor import extract_code

_env = Environment(loader=PackageLoader("rune", "templates"), undefined=StrictUndefined)


def render_template(template_name: str, **kwargs: Any) -> str:
    return _env.get_template(f"{template_name}.j2").render(**kwargs)


class SubtaskSchema(BaseModel):
    name: str
    description: str
    depends_on: list[str] = []


class DecomposeResult(BaseModel):
    subtasks: list[SubtaskSchema]


class DiagnosisEntry(BaseModel):
    subtask_name: str
    error_type: str
    location: str
    fix_guidance: str


class DiagnoseResult(BaseModel):
    entries: list[DiagnosisEntry]


_FIX_GUIDANCE_CAP = 150


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
        case "code":
            target = action.target_subtask
            passed = feedback is not None and feedback.exit_code == 0
            retries = dict(state.get("retries", {}))
            diagnosis = dict(state.get("diagnosis", {}))
            fb_map = dict(state.get("feedback", {}))
            is_resample = target in state.get("code_results", {})
            if is_resample:
                retries[target] = 0
            diagnosis.pop(target, None)
            if feedback:
                fb_map[target] = feedback
            return {
                "code_results": {
                    **state.get("code_results", {}),
                    target: extract_code(raw),
                },
                "code_passed": {**state.get("code_passed", {}), target: passed},
                "retries": retries,
                "feedback": fb_map,
                "diagnosis": diagnosis,
            }
        case "repair":
            target = action.target_subtask
            passed = feedback is not None and feedback.exit_code == 0
            retries = dict(state.get("retries", {}))
            retries[target] = retries.get(target, 0) + 1
            diagnosis = dict(state.get("diagnosis", {}))
            diagnosis.pop(target, None)
            fb_map = dict(state.get("feedback", {}))
            if feedback:
                fb_map[target] = feedback
            return {
                "code_results": {
                    **state.get("code_results", {}),
                    target: extract_code(raw),
                },
                "code_passed": {**state.get("code_passed", {}), target: passed},
                "retries": retries,
                "feedback": fb_map,
                "diagnosis": diagnosis,
            }
        case "integrate":
            passed = feedback is not None and feedback.exit_code == 0
            return {
                "integrated_code": extract_code(raw) if passed else "",
                "integration_feedback": feedback,
                "diagnosis": {},
            }
        case "diagnose":
            diag_result = DiagnoseResult.model_validate_json(raw)
            diagnosis = dict(state.get("diagnosis", {}))
            for entry in diag_result.entries:
                diagnosis[entry.subtask_name] = entry.fix_guidance[:_FIX_GUIDANCE_CAP]
            return {"diagnosis": diagnosis}
    return {}
