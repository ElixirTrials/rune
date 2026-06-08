"""Deterministic plan validation before code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rune.engine.requirements import _expected_param_names

_SINGLE_ARG_PHRASES = re.compile(
    r"\b(single[- ]argument|one parameter|takes one argument|"
    r"function of one argument|only one argument)\b",
    re.IGNORECASE,
)
_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "s": ("string", "input string", "characters"),
    "grid": ("matrix", "grid"),
    "digits": ("digit", "multiset"),
    "nums": ("array", "numbers", "elements"),
    "l": ("lower", "range", "left"),
    "r": ("upper", "right"),
    "k": ("changes", "flips", "flip"),
    "limit": ("product limit", "maximum product", "cap"),
}
_WRONG_DIGITS_PLAN = re.compile(
    r"\b("
    r"100\s*(?:to|-)\s*999|"
    r"0\s*(?:to|-)\s*999|"
    r"10\s*\*\*\s*\(|"
    r"three[- ]digit\s+numbers?\s+from\s+100"
    r")\b",
    re.IGNORECASE,
)
_CONTIGUOUS_WHEN_SUBSEQUENCE = re.compile(
    r"\bcontiguous\s+subsequence\b", re.IGNORECASE
)


@dataclass(frozen=True)
class PlanGateResult:
    """Outcome of validating a subtask plan against task contract."""

    ok: bool
    deficiencies: tuple[str, ...]


def validate_plan(
    plan_text: str,
    *,
    entry_point: str,
    signature: str,
    public_checks: str = "",
    task_spec: str = "",
) -> PlanGateResult:
    """Check plan against entry_point signature (deterministic; fail-open after cap).

    Args:
        plan_text: Prose plan from the plan action.
        entry_point: Expected function name.
        signature: Starter code signature source.
        public_checks: Public oracle asserts (used for arity sanity only).

    Returns:
        PlanGateResult with ``ok=False`` when contract signals are missing.
    """
    text = (plan_text or "").strip()
    deficiencies: list[str] = []

    if not text:
        return PlanGateResult(ok=False, deficiencies=("empty plan",))

    expected_params = (
        _expected_param_names(signature, entry_point) if signature else None
    )
    n_params = len(expected_params) if expected_params else 0
    if entry_point and entry_point not in text:
        has_param_anchor = False
        if expected_params:
            for param in expected_params:
                if _param_mentioned(param, text):
                    has_param_anchor = True
                    break
        if not has_param_anchor:
            deficiencies.append(f"plan does not mention entry_point `{entry_point}`")

    if expected_params:
        for param in expected_params:
            if not _param_mentioned(param, text):
                deficiencies.append(f"plan omits parameter `{param}`")
        if "digits" in expected_params and _WRONG_DIGITS_PLAN.search(text):
            deficiencies.append(
                "plan treats input as a length/range, but `digits` is a list multiset"
            )
    spec_lower = (task_spec or "").lower()
    if (
        "subsequence" in spec_lower
        and "contiguous" not in spec_lower
        and _CONTIGUOUS_WHEN_SUBSEQUENCE.search(text)
    ):
        deficiencies.append(
            "plan uses contiguous subsequence but task asks for subsequence "
            "(not necessarily contiguous)"
        )
        if n_params >= 2 and _SINGLE_ARG_PHRASES.search(text):
            deficiencies.append(
                "plan describes a single-argument function but task requires "
                f"{n_params} parameters"
            )

    return PlanGateResult(ok=not deficiencies, deficiencies=tuple(deficiencies))


def _param_mentioned(param: str, text: str) -> bool:
    if re.search(rf"\b{re.escape(param)}\b", text, re.IGNORECASE):
        return True
    for alias in _PARAM_ALIASES.get(param, ()):
        if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
            return True
    return False


def format_plan_deficiency_feedback(deficiencies: tuple[str, ...]) -> str:
    """Planner-facing message when the plan gate rejects a plan."""
    bullets = "\n".join(f"- {d}" for d in deficiencies)
    return f"Plan rejected — fix and replan:\n{bullets}"
