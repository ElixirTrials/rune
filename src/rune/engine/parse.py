"""Jinja2 template rendering and structured output parsing for engine actions."""

from __future__ import annotations

import logging
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from rune.engine.json_repair import extract_code_value
from rune.engine.state import Action, Feedback, Subtask

logger = logging.getLogger(__name__)

_env = Environment(loader=PackageLoader("rune", "templates"), undefined=StrictUndefined)


def render_template(template_name: str, **kwargs: Any) -> str:
    return _env.get_template(f"{template_name}.j2").render(**kwargs)


class SubtaskSchema(BaseModel):
    name: str
    description: str
    depends_on: list[str] = []


class DecomposeResult(BaseModel):
    subtasks: list[SubtaskSchema]


class PlanResult(BaseModel):
    plan: str


class CodeResult(BaseModel):
    code: str


class IntegrateResult(BaseModel):
    code: str


class DiagnosisEntry(BaseModel):
    subtask_name: str
    error_type: str
    location: str
    fix_guidance: str


class DiagnoseResult(BaseModel):
    entries: list[DiagnosisEntry]


_FIX_GUIDANCE_CAP = 150


def _parse_code_action(
    target: str | None,
    raw: str,
    feedback: Feedback | None,
    state: dict[str, Any],
    *,
    retries_delta: int,
) -> dict[str, Any]:
    try:
        code = CodeResult.model_validate_json(raw).code
    except Exception:
        code = extract_code_value(raw)
    passed = feedback is not None and feedback.exit_code == 0
    retries = dict(state.get("retries", {}))
    retries[target] = retries.get(target, 0) + retries_delta
    diagnosis = dict(state.get("diagnosis", {}))
    diagnosis.pop(target, None)
    fb_map = dict(state.get("feedback", {}))
    if feedback:
        fb_map[target] = feedback
    return {
        "code_results": {
            **state.get("code_results", {}),
            target: code,
        },
        "code_passed": {
            **state.get("code_passed", {}),
            target: passed,
        },
        "retries": retries,
        "feedback": fb_map,
        "diagnosis": diagnosis,
    }


def parse_output(
    action: Action,
    raw: str,
    feedback: Feedback | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    match action.name:
        case "decompose":
            try:
                result = DecomposeResult.model_validate_json(raw)
            except Exception:
                logger.warning(
                    "decompose output failed validation; re-decomposing"
                )
                return {}
            names = {s.name for s in result.subtasks}
            return {
                "subtasks": [
                    Subtask(
                        name=s.name,
                        description=s.description,
                        # Drop phantom (typo'd/unknown) and self dependencies so
                        # readiness checks and the DAG can never softlock.
                        depends_on=[
                            d for d in s.depends_on if d in names and d != s.name
                        ],
                    )
                    for s in result.subtasks
                ]
            }
        case "plan":
            target = action.target_subtask
            try:
                plan_text = PlanResult.model_validate_json(raw).plan
            except Exception:
                logger.warning(
                    "plan output failed validation for %s; re-planning", target
                )
                return {}
            return {"plans": {**state.get("plans", {}), target: plan_text}}
        case "code":
            target = action.target_subtask
            # The first code attempt for a target is not a retry; only resamples
            # (code re-issued after repairs exhausted) and repairs count.
            first_attempt = target not in state.get("code_results", {})
            return _parse_code_action(
                target,
                raw,
                feedback,
                state,
                retries_delta=0 if first_attempt else 1,
            )
        case "repair":
            return _parse_code_action(
                action.target_subtask,
                raw,
                feedback,
                state,
                retries_delta=1,
            )
        case "integrate":
            try:
                code = IntegrateResult.model_validate_json(raw).code
            except Exception:
                code = extract_code_value(raw)
            passed = feedback is not None and feedback.exit_code == 0
            return {
                "integrated_code": code if passed else "",
                "integration_feedback": feedback,
                "diagnosis": {},
            }
        case "diagnose":
            try:
                diag_result = DiagnoseResult.model_validate_json(raw)
            except Exception:
                logger.warning(
                    "diagnose output failed validation; no diagnosis recorded"
                )
                return {}
            diagnosis = dict(state.get("diagnosis", {}))
            code_passed = dict(state.get("code_passed", {}))
            for entry in diag_result.entries:
                diagnosis[entry.subtask_name] = entry.fix_guidance[:_FIX_GUIDANCE_CAP]
                # Re-open diagnosed subtasks so select_action routes them to
                # repair. Without this an integration-failure diagnose (which
                # leaves every code_passed True) never triggers a repair and the
                # engine livelocks integrate<->diagnose until budget is spent.
                if entry.subtask_name in code_passed:
                    code_passed[entry.subtask_name] = False
            return {"diagnosis": diagnosis, "code_passed": code_passed}
    logger.warning(
        "Unknown action %r in parse_output, returning empty update",
        action.name,
    )
    return {}
