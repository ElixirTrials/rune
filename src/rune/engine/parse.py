"""Jinja2 template rendering and structured output parsing for engine actions."""

from __future__ import annotations

import logging
import re
from typing import Any

import json_repair
from jinja2 import Environment, PackageLoader, StrictUndefined
from markdown_it import MarkdownIt
from pydantic import BaseModel

from rune.engine.state import Action, Feedback, Subtask

logger = logging.getLogger(__name__)

# CommonMark parser for code-block extraction (markdown-it-py, the reference
# parser used by rich/mkdocs/jupyter) — robust where regex is fragile.
_MARKDOWN = MarkdownIt("commonmark")

_env = Environment(
    loader=PackageLoader("rune", "templates"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


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


# Subtasks that are project chores, not implementation units. The model tends
# to split trivial single-function tasks into these, inflating step counts and
# the integration-failure surface. Dropped at decompose-time (but never if it
# would empty the plan).
_CHORE_RE = re.compile(
    r"\b("
    r"documentation|docstrings?|"
    r"unit tests?|write tests?|add tests?|test cases?|testing|"
    r"edge cases?|"
    r"function signature|type hints?|annotations?|"
    r"comments?"
    r")\b",
    re.IGNORECASE,
)


def _is_chore_subtask(s: SubtaskSchema) -> bool:
    # Match the NAME only. Matching the description would drop legitimate
    # implementation subtasks whose subject is docs/tests/annotations
    # (e.g. a docstring parser, a type-hint linter) — a false positive that
    # would silently nuke real work on every `rune run`. Conservative by design.
    return bool(_CHORE_RE.search(s.name))


_FIX_GUIDANCE_CAP = 150


def _extract_code_block(value: str) -> str:
    """Return the first fenced code block's content, else *value* unchanged.

    Even under xgrammar-constrained structured output the instruct model wraps
    its code in a ```lang ... ``` fence *inside* the JSON ``code`` string
    (verified: ``result.text == '{"code": "```py\\n..."}'``); the leading fence
    line then crashes the sandbox with ``SyntaxError`` on line 1, turning a
    correct solution into a spurious failure (and a wasted repair loop).

    Extraction uses the CommonMark tokenizer (``markdown-it-py``), not regex:
    it handles ```lang info strings, an *unterminated* fence (truncated output
    becomes a fence to EOF), and passes bare code (no fence) through unchanged.
    Only explicit ``` fences match — *not* indented ``code_block`` tokens, which
    would mis-extract a real Python body after a blank line.
    """
    for tok in _MARKDOWN.parse(value):
        if tok.type == "fence":
            return tok.content.rstrip("\n")
    return value


def extract_code_from_raw(
    raw: str, model: type[BaseModel], *, fallback_to_raw: bool = False
) -> str:
    """Parse *raw* as *model*'s ``code`` field, de-fenced.

    Pipeline (maintained libraries, no fragile regex): ``json-repair`` robustly
    parses the model's JSON (repairing truncation / prose-wrapping) → ``model``
    (pydantic) validates the structure → ``markdown-it-py`` extracts the code
    from the possibly-fenced ``code`` value. A non-pydantic path is logged
    loudly — it is a signal, not silent behavior. ``fallback_to_raw`` de-fences
    the raw text only when no structured ``code`` can be recovered at all.
    """
    obj: Any = None
    try:
        obj = json_repair.loads(raw)
    except Exception:  # pragma: no cover - json_repair is very tolerant
        obj = None

    if isinstance(obj, dict):
        try:
            code: str = model.model_validate(obj).code  # type: ignore[attr-defined]
            return _extract_code_block(code)
        except Exception:
            recovered = obj.get("code")
            if isinstance(recovered, str):
                logger.warning(
                    "%s: json-repair recovered a code field that failed full "
                    "schema validation; using it",
                    model.__name__,
                )
                return _extract_code_block(recovered)

    if fallback_to_raw:
        logger.warning(
            "%s: could not parse structured output even with json-repair; "
            "de-fencing raw text (pydantic bypassed)",
            model.__name__,
        )
        return _extract_code_block(raw)
    return ""


def _parse_code_action(
    target: str | None,
    raw: str,
    feedback: Feedback | None,
    state: dict[str, Any],
    *,
    retries_delta: int,
    code: str | None = None,
) -> dict[str, Any]:
    if code is None:
        code = extract_code_from_raw(raw, CodeResult)
    passed = feedback is not None and feedback.exit_code == 0
    retries = dict(state.get("retries", {}))
    retries[target] = retries.get(target, 0) + retries_delta
    diagnosis = dict(state.get("diagnosis", {}))
    diagnosis.pop(target, None)
    fb_map = dict(state.get("feedback", {}))
    if feedback:
        fb_map[target] = feedback
    result: dict[str, Any] = {
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
    subtasks = state.get("subtasks", [])
    if passed and len(subtasks) == 1:
        result["integrated_code"] = code
    return result


def parse_output(
    action: Action,
    raw: str,
    feedback: Feedback | None,
    state: dict[str, Any],
    *,
    code: str | None = None,
) -> dict[str, Any]:
    # ``code``, when provided, is the already-extracted code the sandbox actually
    # ran (from graph.step_node); using it keeps state's recorded code identical
    # to what executed instead of re-parsing raw with a divergent fallback.
    match action.name:
        case "decompose":
            try:
                result = DecomposeResult.model_validate_json(raw)
            except Exception:
                logger.warning("decompose output failed validation; re-decomposing")
                return {}
            # Drop pure-chore subtasks (docs/tests/edge-cases/signatures) — but
            # never empty the plan; degrade to keeping everything if all are chores.
            kept = [s for s in result.subtasks if not _is_chore_subtask(s)]
            if not kept:
                kept = list(result.subtasks)
            names = {s.name for s in kept}
            return {
                "subtasks": [
                    Subtask(
                        name=s.name,
                        description=s.description,
                        # Drop phantom (typo'd/unknown), self, and dropped-chore
                        # dependencies so readiness checks and the DAG never softlock.
                        depends_on=[
                            d for d in s.depends_on if d in names and d != s.name
                        ],
                    )
                    for s in kept
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
                code=code,
            )
        case "repair":
            return _parse_code_action(
                action.target_subtask,
                raw,
                feedback,
                state,
                retries_delta=1,
                code=code,
            )
        case "integrate":
            if code is None:
                code = extract_code_from_raw(raw, IntegrateResult)
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
            reopened = False
            for entry in diag_result.entries:
                diagnosis[entry.subtask_name] = entry.fix_guidance[:_FIX_GUIDANCE_CAP]
                # Re-open diagnosed subtasks so select_action routes them to
                # repair. Without this an integration-failure diagnose (which
                # leaves every code_passed True) never triggers a repair and the
                # engine livelocks integrate<->diagnose until budget is spent.
                if entry.subtask_name in code_passed:
                    code_passed[entry.subtask_name] = False
                    reopened = True
            # Targeted diagnose: the model often invents a subtask_name (e.g.
            # "write_function" for the real "_main"), so the per-entry write
            # above lands on a phantom key and select_action never sees a
            # diagnosis for the target — the engine re-diagnoses every step until
            # the budget is spent, never reaching repair. There is exactly one
            # target, so attach the model's guidance to it regardless of naming.
            target = action.target_subtask
            if target is not None and target not in diagnosis:
                guidance = (
                    "; ".join(e.fix_guidance for e in diag_result.entries).strip()[
                        :_FIX_GUIDANCE_CAP
                    ]
                    or "revise this subtask"
                )
                diagnosis[target] = guidance
                if target in code_passed:
                    code_passed[target] = False

            # Untargeted (integration-failure) diagnose: the model often emits
            # subtask_name values that don't match any real subtask, so the
            # per-entry reopen above is a no-op and the engine livelocks. When
            # nothing matched, deterministically reopen every subtask (with the
            # model's guidance) so they route to repair regardless of naming.
            if action.target_subtask is None and not reopened and code_passed:
                guidance = (
                    "; ".join(e.fix_guidance for e in diag_result.entries).strip()[
                        :_FIX_GUIDANCE_CAP
                    ]
                    or "integration failed; revise this subtask"
                )
                for name in code_passed:
                    code_passed[name] = False
                    diagnosis.setdefault(name, guidance)
            return {"diagnosis": diagnosis, "code_passed": code_passed}
    logger.warning(
        "Unknown action %r in parse_output, returning empty update",
        action.name,
    )
    return {}
