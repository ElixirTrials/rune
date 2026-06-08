"""Backward-compatible wrappers around the task requirements oracle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rune.engine.requirements import (
    RequirementContext,
    evaluate_task_requirements,
    format_requirements_feedback,
)


@dataclass(frozen=True)
class ValidityResult:
    """Outcome of the solution validity gate."""

    ok: bool
    deficiencies: tuple[str, ...]


def format_validity_feedback(deficiencies: tuple[str, ...]) -> str:
    """Repair-facing message listing each failed check explicitly."""
    return format_requirements_feedback(deficiencies)


def validate_solution(
    code: str,
    *,
    entry_point: str,
    signature: str,
    spec: str,
    public_checks: str,
) -> ValidityResult:
    """Run task requirements; active only when benchmark public_checks are wired."""
    ctx = RequirementContext(
        entry_point=entry_point,
        signature=signature,
        spec=spec,
        public_checks=public_checks,
    )
    ok, deficiencies = evaluate_task_requirements(code, ctx)
    return ValidityResult(ok=ok, deficiencies=deficiencies)


def validate_state_code(state: Mapping[str, Any], code: str) -> ValidityResult:
    """Validate using fields from RunState."""
    ctx = RequirementContext.from_state(state)
    ok, deficiencies = evaluate_task_requirements(code, ctx)
    return ValidityResult(ok=ok, deficiencies=deficiencies)
