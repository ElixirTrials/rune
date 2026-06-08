"""Jinja2 template rendering and structured output parsing for engine actions."""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

import json_repair
from jinja2 import Environment, PackageLoader, StrictUndefined
from markdown_it import MarkdownIt
from pydantic import BaseModel, field_validator

from rune.engine.oracle import defines_entry_point
from rune.engine.requirements import is_constraint_scale_only_failure
from rune.engine.state import Action, Feedback, Subtask
from rune.engine.validity import validate_state_code

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
    # Episodic design: each subtask carries concrete acceptance checks (2–4
    # distinct asserts covering different behaviors) and names the piece of the
    # final entry_point it builds — so each subtask runs a real dev cycle and
    # integration can be AST-verified.
    acceptance_check: str = ""
    builds: str = ""

    @field_validator("acceptance_check", mode="before")
    @classmethod
    def _normalize_acceptance_check(cls, value: object) -> str:
        # Accept a JSON list of asserts or a single string. The oracle parses the
        # result via the AST (tolerant of the over-escaped form), so no textual
        # fix-ups here.
        if isinstance(value, list):
            return "\n".join(str(item).strip() for item in value if str(item).strip())
        return str(value) if value is not None else ""


class DecomposeResult(BaseModel):
    subtasks: list[SubtaskSchema]
    # Condensed overall goal for the episodic adapter (NOT the full spec at every
    # step). For a single-function task the lone subtask carries the full task.
    overall_goal: str = ""


_DECOMPOSE_MAX_SUBTASKS = 3


def _loads_structured[M: BaseModel](raw: str, model: type[M]) -> M | None:
    """Parse *raw* as *model* JSON, recovering from truncation/prose-wrapping.

    Plain ``model_validate_json`` fails hard on a truncated or lightly-malformed
    JSON object (long hard-task output) — which previously caused a re-plan /
    re-decompose loop. ``json_repair`` repairs the JSON first; on unrecoverable
    input returns ``None`` so callers can degrade gracefully instead of looping.
    """
    try:
        return model.model_validate_json(raw)
    except Exception:  # noqa: BLE001 - any malformed JSON; try repair next
        pass
    try:
        obj = json_repair.loads(raw)
    except Exception:  # pragma: no cover - json_repair is very tolerant
        return None
    if isinstance(obj, dict):
        try:
            return model.model_validate(obj)
        except Exception:  # noqa: BLE001 - repaired but still not the schema
            return None
    return None


def _fn_from_check(check: str) -> str:
    """The function name a subtask's ``acceptance_check`` calls — the authoritative
    name for the function the subtask must define (so prompt/code/check/test agree).

    Returns the first ``Name``-callee in the check (e.g. ``assert tokenize('2+3')
    == [...]`` -> ``"tokenize"``), or ``""`` if the check is empty/unparseable.
    """
    if not check.strip():
        return ""
    try:
        tree = ast.parse(check.strip())
    except (SyntaxError, ValueError):
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
    return ""


def _force_check_calls(check: str, old_name: str, new_name: str) -> str:
    """Rewrite calls to ``old_name`` in *check* so they call ``new_name`` instead.

    Holds the subtask name and its ``acceptance_check`` deterministically
    consistent: after a subtask is (re)named (e.g. forced to the entry_point),
    its check must call that SAME name — otherwise the in-loop probe runs a call
    to a function the code never defines and manufactures a spurious NameError
    (issue #52). Only the function under test (``old_name``) is rewritten; builtins
    and other helpers in the check are left untouched.
    """
    if not new_name or old_name == new_name or not check.strip():
        return check
    try:
        tree = ast.parse(check.strip())
    except (SyntaxError, ValueError):
        return check

    class _Rename(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.Call:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == old_name:
                node.func.id = new_name
            return node

    try:
        return ast.unparse(_Rename().visit(tree))
    except (ValueError, AttributeError):
        return check


def _collapse_benchmark_subtasks(
    kept: list[SubtaskSchema],
    state: dict[str, Any],
    overall_goal: str,
) -> list[SubtaskSchema]:
    """Force one entry_point subtask when LCB/MBPP public_checks are wired.

    Decompose often fans out into misnamed helpers; benchmark tasks always grade
    a single top-level ``entry_point`` function, so subtask names and acceptance
    checks must match it deterministically.
    """
    entry_pt = str(state.get("entry_point", "") or "")
    public = str(state.get("public_checks", "") or "").strip()
    if not entry_pt or not public:
        return kept
    desc = overall_goal.strip()
    if not desc:
        desc = next(
            (s.description for s in kept if s.name == entry_pt and s.description),
            "",
        )
    if not desc:
        desc = str(state.get("task", ""))[:500]
    return [
        SubtaskSchema(
            name=entry_pt,
            description=desc,
            acceptance_check=public,
            builds=entry_pt,
            depends_on=[],
        )
    ]


def _single_subtask_fallback(state: dict[str, Any]) -> dict[str, Any]:
    """One whole-task subtask (used when decompose is unparseable).

    Keeps the engine moving instead of re-decompose-looping: the sole subtask is
    the full task, named for the entry_point so integration verification can find
    it. Its acceptance check is the task's public example(s), surfaced by the
    oracle from the spec — so it still gets a real in-loop signal.
    """
    task = str(state.get("task", ""))
    entry = str(state.get("entry_point", "")) or "main"
    public = str(state.get("public_checks", "") or "").strip()
    return {
        "overall_goal": task[:200],
        "subtasks": [
            Subtask(
                name=entry,
                description=task,
                depends_on=[],
                acceptance_check=public,
                builds=entry,
            )
        ],
    }


class PlanResult(BaseModel):
    plan: str


class ComplexityJudgeResult(BaseModel):
    """Adapter-backed static complexity assessment when empirical big_o times out.

    Field order matches :class:`JudgeResult`: analysis before the verdict commit.
    """

    reason: str = ""
    measured_complexity: str = ""
    sufficient: bool = True


class JudgeResult(BaseModel):
    """Model correctness verdict on a candidate implementation.

    Field order matters: structured output is emitted in declaration order, so
    ``reason`` (the analysis) comes BEFORE the ``correct`` verdict — committing to
    a verdict first made the model guess then rationalise, producing false
    positives on correct code (e.g. naming "4" for a correct int_to_roman while the
    reasoning concluded it was fine). ``correct=False`` must be grounded by a
    concrete ``failing_input``; an ungrounded "looks wrong" is treated as correct,
    to avoid false-positive repairs on already-correct code.
    """

    reason: str = ""
    failing_input: str = ""
    correct: bool


class DiagnosisEntry(BaseModel):
    subtask_name: str
    error_type: str
    location: str = ""
    fix_guidance: str
    violated_invariant: str = ""
    observed_vs_expected: str = ""


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


_FIX_GUIDANCE_CAP = 500


def extract_code_block(value: str) -> str:
    """Return the first fenced code block's content, else *value* unchanged.

    Code actions emit freeform Python — a ```lang ... ``` fence (the instruct
    model's natural format) or bare code — never a JSON ``{"code": ...}`` object.
    Wrapping code in a JSON string let the model over-escape newlines
    (``\\n`` -> literal backslash-n) and collapse multi-line code to one line, a
    phantom ``SyntaxError`` on line 1. De-fencing the raw output sidesteps that
    class entirely.

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


def candidate_quality(
    code: str,
    feedback: Feedback | None,
    *,
    constraint_scale_pass_quality: bool = True,
) -> int:
    """Rank a code candidate by its sandbox outcome (higher = better to ship).

    3 = passed the in-loop check (or visible-correct but advisory slow);
    2 = ran but mismatched (AssertionError — a near-miss that still executes);
    1 = compiled but crashed at runtime; 0 = empty or a syntax error. Used to
    keep the BEST candidate per subtask so a later worse attempt can't be the
    one shipped (issue #52 RC-C).
    """
    if not (code or "").strip():
        return 0
    if feedback is not None and feedback.exit_code == 0:
        return 3
    stderr = feedback.stderr if feedback is not None else ""
    if constraint_scale_pass_quality and is_constraint_scale_only_failure(stderr):
        return 3
    if "SyntaxError" in stderr:
        return 0
    if "AssertionError" in stderr:
        return 2
    return 1


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
        code = extract_code_block(raw)
    passed = feedback is not None and feedback.exit_code == 0
    retries = dict(state.get("retries", {}))
    retries[target] = retries.get(target, 0) + retries_delta
    diagnosis = dict(state.get("diagnosis", {}))
    diagnosis.pop(target, None)
    fb_map = dict(state.get("feedback", {}))
    if feedback:
        fb_map[target] = feedback
    # No-regress: retain the highest-quality candidate seen for this subtask so a
    # later re-code/repair can't throw away a near-miss by shipping a crash.
    quality = candidate_quality(
        code,
        feedback,
        constraint_scale_pass_quality=bool(
            state.get("constraint_scale_pass_quality", True)
        ),
    )
    best_code = dict(state.get("best_code", {}))
    best_quality = dict(state.get("best_quality", {}))
    ship_ok = True
    if passed and str(state.get("public_checks", "") or "").strip():
        ship_ok = validate_state_code(state, code).ok
    if ship_ok and quality >= best_quality.get(target, -1):
        best_code[target] = code
        best_quality[target] = quality
    code_solved = dict(state.get("code_solved", {}))
    if passed:
        code_solved[target] = True
    result: dict[str, Any] = {
        "code_results": {
            **state.get("code_results", {}),
            target: code,
        },
        "code_passed": {
            **state.get("code_passed", {}),
            target: passed,
        },
        "code_solved": code_solved,
        "best_code": best_code,
        "best_quality": best_quality,
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
            result = _loads_structured(raw, DecomposeResult)
            if result is None or not result.subtasks:
                # Degrade to ONE whole-task subtask rather than returning {} (which
                # re-decompose-loops and burns budget to empty code).
                logger.warning("decompose unparseable; degrading to single subtask")
                return _single_subtask_fallback(state)
            # Drop pure-chore subtasks (docs/tests/edge-cases/signatures) — but
            # never empty the plan; degrade to keeping everything if all are chores.
            kept = [s for s in result.subtasks if not _is_chore_subtask(s)]
            if not kept:
                kept = list(result.subtasks)
            kept = _collapse_benchmark_subtasks(kept, state, result.overall_goal)
            entry_pt = str(state.get("entry_point", ""))
            # Authoritative name transmission: a subtask's function name is the
            # function its acceptance_check CALLS (what the model committed to
            # testing) — never a descriptive phrase. This keeps the thin prompt
            # ("implement <name>"), the generated code, the acceptance_check, and
            # the held-out test (which calls the entry_point) all naming the SAME
            # function — the source of the earlier NameError / duplicate-name churn.
            for s in kept:
                fn = _fn_from_check(s.acceptance_check)
                if fn:
                    s.name = fn
                    s.builds = s.builds or fn
            # A subtask whose function IS the entry_point is the whole task -> keep
            # only it (drops redundant helpers / duplicate entry_point subtasks).
            if entry_pt and any(s.name == entry_pt for s in kept):
                kept = [s for s in kept if s.name == entry_pt][:1]
            # Collapse duplicate names (the model sometimes emits the same subtask
            # N times -> N× the work and a guaranteed exhaust). Keep first of each.
            _seen: set[str] = set()
            _deduped: list[SubtaskSchema] = []
            for s in kept:
                if s.name in _seen:
                    continue
                _seen.add(s.name)
                _deduped.append(s)
            kept = _deduped
            kept = kept[:_DECOMPOSE_MAX_SUBTASKS]  # bound (owner: cap <=3)
            # Single-function task: the entry_point is the authoritative name (the
            # held-out test calls it), so the lone subtask must define exactly it.
            if len(kept) == 1 and entry_pt:
                kept[0].name = entry_pt
                kept[0].builds = entry_pt
            public_checks = str(state.get("public_checks", "") or "").strip()
            if (
                public_checks
                and len(kept) == 1
                and entry_pt
                and kept[0].name == entry_pt
            ):
                kept[0].acceptance_check = public_checks
            # Deterministic name<->check consistency: every kept subtask's
            # acceptance_check must call its own (now-final) name, so the thin
            # prompt, the generated code, the in-loop check, and the held-out test
            # all reference the SAME function and the name issue can't recur (#52).
            for s in kept:
                s.acceptance_check = _force_check_calls(
                    s.acceptance_check, _fn_from_check(s.acceptance_check), s.name
                )
            names = {s.name for s in kept}
            return {
                "overall_goal": result.overall_goal,
                "subtasks": [
                    Subtask(
                        name=s.name,
                        description=s.description,
                        # Drop phantom (typo'd/unknown), self, and dropped-chore
                        # dependencies so readiness checks and the DAG never softlock.
                        depends_on=[
                            d for d in s.depends_on if d in names and d != s.name
                        ],
                        acceptance_check=s.acceptance_check,
                        builds=s.builds,
                    )
                    for s in kept
                ],
            }
        case "plan":
            target = action.target_subtask
            res = _loads_structured(raw, PlanResult)
            if res is None:
                # Degrade to a minimal plan rather than re-planning to empty.
                logger.warning("plan unparseable for %s; using minimal plan", target)
                plan_text = "Implement the function directly to satisfy the task."
            else:
                plan_text = res.plan
            if state.get("plan_gate_enabled", True) and target:
                from rune.engine.plan_gate import (  # noqa: PLC0415
                    format_plan_deficiency_feedback,
                    validate_plan,
                )

                gate = validate_plan(
                    plan_text,
                    entry_point=str(state.get("entry_point", "") or ""),
                    signature=str(state.get("signature", "") or ""),
                    public_checks=str(state.get("public_checks", "") or ""),
                    task_spec=str(state.get("task", "") or ""),
                )
                if not gate.ok:
                    attempts = dict(state.get("plan_attempts", {}))
                    n = attempts.get(target, 0) + 1
                    attempts[target] = n
                    rejections = {
                        **state.get("plan_rejections", {}),
                        target: format_plan_deficiency_feedback(gate.deficiencies),
                    }
                    max_attempts = int(state.get("plan_gate_max_attempts", 2))
                    if n < max_attempts:
                        logger.info(
                            "plan gate rejected %s (attempt %d): %s",
                            target,
                            n,
                            gate.deficiencies,
                        )
                        return {
                            "plan_attempts": attempts,
                            "plan_rejections": rejections,
                        }
                    logger.warning(
                        "plan gate bypassed for %s after %d attempts",
                        target,
                        n,
                    )
            replan_targets = dict(state.get("replan_targets", {}))
            replan_targets.pop(str(target), None)
            plan_rejections = dict(state.get("plan_rejections", {}))
            plan_rejections.pop(str(target), None)
            return {
                "plans": {**state.get("plans", {}), target: plan_text},
                "replan_targets": replan_targets,
                "plan_rejections": plan_rejections,
            }
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
                code = extract_code_block(raw)
            entry_pt = str(state.get("entry_point", ""))
            sandbox_ok = feedback is not None and feedback.exit_code == 0
            # Verified (owner: "bounded + verified"): integration must actually
            # DEFINE the entry_point (AST), not merely run without crashing.
            defines = (not entry_pt) or defines_entry_point(code, entry_pt)
            passed = sandbox_ok and defines
            return {
                "integrated_code": code if passed else "",
                "integration_feedback": feedback,
                "diagnosis": {},
            }
        case "diagnose":
            from rune.engine.repair_brief import (  # noqa: PLC0415
                merge_guidance_with_brief,
            )

            diag_result = _loads_structured(raw, DiagnoseResult)
            if diag_result is None:
                logger.warning("diagnose output unparseable; no diagnosis recorded")
                return {}
            diagnosis = dict(state.get("diagnosis", {}))
            code_passed = dict(state.get("code_passed", {}))
            reopened = False
            target = action.target_subtask
            brief_text = state.get("repair_briefs", {}).get(str(target or ""), "")
            code_solved = state.get("code_solved", {})
            for entry in diag_result.entries:
                diagnosis[entry.subtask_name] = merge_guidance_with_brief(
                    brief_text,
                    entry.fix_guidance,
                    llm_failure_class=entry.error_type,
                    llm_violated_invariant=entry.violated_invariant,
                    llm_observed_vs_expected=entry.observed_vs_expected,
                )[:_FIX_GUIDANCE_CAP]
                # Re-open diagnosed subtasks so select_action routes them to
                # repair. Without this an integration-failure diagnose (which
                # leaves every code_passed True) never triggers a repair and the
                # engine livelocks integrate<->diagnose until budget is spent.
                if entry.subtask_name in code_passed and not code_solved.get(
                    entry.subtask_name
                ):
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
                if target in code_passed and not code_solved.get(target):
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
                    if code_solved.get(name):
                        continue
                    code_passed[name] = False
                    diagnosis.setdefault(name, guidance)
            return {"diagnosis": diagnosis, "code_passed": code_passed}
    logger.warning(
        "Unknown action %r in parse_output, returning empty update",
        action.name,
    )
    return {}
