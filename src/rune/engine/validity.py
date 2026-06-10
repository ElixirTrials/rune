"""Backward-compatible wrappers around the task requirements oracle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rune.config import PipelineConfig
from rune.engine.complexity import ComplexityProbeConfig
from rune.engine.requirements import (
    RequirementContext,
    evaluate_task_requirements,
    format_requirements_feedback,
)
from rune.engine.state import advisory_kinds_from_state


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
    skip_kinds: frozenset[str] = frozenset(),
) -> ValidityResult:
    """Run task requirements; active only when benchmark public_checks are wired."""
    defaults = PipelineConfig()
    ctx = RequirementContext(
        entry_point=entry_point,
        signature=signature,
        spec=spec,
        public_checks=public_checks,
        complexity_probe=ComplexityProbeConfig(
            min_n=defaults.complexity_probe_min_n,
            max_n=defaults.complexity_probe_max_n,
            n_repeats=defaults.complexity_probe_n_repeats,
            per_run_timeout_s=defaults.complexity_probe_per_run_timeout_s,
        ),
    )
    ok, deficiencies = evaluate_task_requirements(code, ctx, skip_kinds=skip_kinds)
    return ValidityResult(ok=ok, deficiencies=deficiencies)


def validate_state_code(state: Mapping[str, Any], code: str) -> ValidityResult:
    """Validate for best-code retention; advisory performance probes are ignored."""
    ctx = RequirementContext.from_state(state)
    ok, deficiencies = evaluate_task_requirements(
        code, ctx, skip_kinds=advisory_kinds_from_state(state)
    )
    return ValidityResult(ok=ok, deficiencies=deficiencies)
