"""Single-node LangGraph engine: step_node + should_continue loop."""

from __future__ import annotations

import ast
import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import mlflow
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rune.bench.lcb import extract_entry_function
from rune.engine.complexity import (
    COMPLEXITY_ANALYSIS_RUBRIC,
    ComplexityProbeConfig,
    ScaleProbeOutcome,
    allowed_complexity_for_max_n,
    build_complexity_assessment_task,
    check_constraint_scale_guarded,
    constraint_max_n,
    constraint_scale_required,
    extract_constraints_block,
    parse_task_constraints,
    static_complexity_signals,
)
from rune.engine.continuation import (
    CONT_SYSTEM_PROMPT,
    degeneration_score,
    extract_partial_code,
    strip_self_tests,
    validate_syntax,
)
from rune.engine.delivery import format_delivery_contract
from rune.engine.oracle import (
    build_probe,
    build_subtask_probe,
    split_acceptance_checks,
)
from rune.engine.parse import (
    ComplexityJudgeResult,
    JudgeResult,
    parse_output,
    render_template,
)
from rune.engine.policy import select_action
from rune.engine.requirements import (
    evaluate_state_requirements,
    format_requirements_feedback,
)
from rune.engine.state import (
    _CODE_HISTORY_CAP,
    Action,
    Feedback,
    RunState,
    StepRecord,
)
from rune.model.adapter import apply_episodic_adapter
from rune.sandbox.executor import run_in_sandbox

logger = logging.getLogger(__name__)

_ORACLE_FAIL_CLOSED_MSG = "oracle checks configured but probe did not fire"


def resolve_in_loop_check(
    name: str, subtask_check: str, state: Mapping[str, Any]
) -> str:
    """Pick in-loop assert source: task public_checks, subtask check, or none."""
    public = str(state.get("public_checks", "") or "").strip()
    entry = str(state.get("entry_point", "") or "")
    sub = subtask_check.strip()
    if public and (not name or name == entry):
        return public
    if sub:
        return sub
    if public:
        return public
    return ""


def _normalize_probe_code(code: str, entry_point: str) -> str:
    if not entry_point or not code.strip():
        return code
    extracted = extract_entry_function(code, entry_point)
    return extracted if extracted.strip() else code


def build_code_probe(
    name: str, code: str, state: Mapping[str, Any]
) -> tuple[str, bool, bool]:
    """Return ``(probe_code, oracle_fired, check_resolved)`` for sandbox execution."""
    stripped = strip_self_tests(code)
    entry_point = str(state.get("entry_point", "") or "")
    normalized = _normalize_probe_code(stripped, entry_point)
    subtask_check = ""
    if name:
        subtask_check = next(
            (s.acceptance_check for s in state.get("subtasks", []) if s.name == name),
            "",
        )
    check = resolve_in_loop_check(name, subtask_check, state)
    check_resolved = bool(check.strip())
    if check_resolved:
        probe, fired = build_subtask_probe(normalized, check)
        return probe, fired, True
    spec = str(state.get("task", "") or "")
    probe, fired = build_probe(normalized, spec, entry_point)
    return probe, fired, False


def apply_oracle_fail_closed(
    probe_fired: bool,
    check_resolved: bool,
    feedback: Feedback,
) -> Feedback:
    """Force failure when checks were configured but the probe could not run them."""
    if check_resolved and not probe_fired and feedback.exit_code == 0:
        return Feedback(
            stdout=feedback.stdout,
            stderr=_ORACLE_FAIL_CLOSED_MSG,
            exit_code=1,
        )
    return feedback


_TARGETED_ACTIONS = frozenset({"plan", "code", "repair"})
# Thin, focused prompts for episodic mode (the adapter carries the context).
# decompose keeps its existing concise prompt (already spec-free).
_EPISODIC_PROMPTS = {
    "plan": "prompt_episodic_plan",
    "code": "prompt_episodic_code",
    "repair": "prompt_episodic_repair",
    "integrate": "prompt_episodic_integrate",
    "decompose": "prompt_decompose_concise",
}
_INTEGRATION_DOC_LINE_CAP = 200
_PROJECT_CAP = 1200
# Carries the task spec into the minimal generation prompt; must NOT truncate a
# short spec mid-example (a 200-char cap cut MBPP docstring asserts off).
_PROJECT_LABEL_CAP = 1200
_ACCUMULATED_CODE_CAP = 3500
# Cap on prior-attempt history folded into the adapter conditioning (R2).
_ATTEMPT_HISTORY_CAP = 3
_ATTEMPT_CODE_CAP = 400
_ATTEMPT_ERR_CAP = 300


def _approach_signature(code: str) -> str:
    """Distinguishing line of an attempt (its return expression) for repair recall.

    The raw stderr is often identical across rounds (same single public assert),
    so listing it alone hides that the model is re-submitting equivalent code
    (issue #52, q3753 steps 6/8/10). Surfacing the return expression lets the
    model see which approaches it already tried.
    """
    returns = [
        ln.strip() for ln in code.splitlines() if ln.strip().startswith("return ")
    ]
    if returns:
        return returns[-1][:80]
    body = [ln.strip() for ln in code.splitlines() if ln.strip()]
    return body[-1][:80] if body else ""


def _format_tried_and_failed(trajectory: list[dict[str, Any]]) -> str:
    """Compact summary of failed repair attempts for episodic recall."""
    lines: list[str] = []
    for entry in trajectory:
        if entry.get("passed"):
            continue
        err = (entry.get("error") or "").strip()
        if not err:
            continue
        step = entry.get("step", "?")
        action = entry.get("action", "attempt")
        snippet = err.splitlines()[-1][:120]
        approach = _approach_signature(entry.get("code") or "")
        detail = f"`{approach}` -> {snippet}" if approach else snippet
        lines.append(f"- step {step} ({action}): {detail}")
    if not lines:
        return ""
    header = (
        "## approaches already tried (all failed public oracle)\n"
        "Do NOT retry these. Try a structurally different algorithm."
    )
    return header + "\n" + "\n".join(lines[-3:])


def _format_attempts(attempts: list[dict[str, Any]] | None) -> str:
    """Render prior (failed) code attempts + their errors for the adapter (R2)."""
    if not attempts:
        return ""
    blocks: list[str] = []
    for i, a in enumerate(attempts[-_ATTEMPT_HISTORY_CAP:], 1):
        code = (a.get("code") or "")[:_ATTEMPT_CODE_CAP]
        err = (a.get("error") or "")[:_ATTEMPT_ERR_CAP]
        blocks.append(f"### Attempt {i}\n{code}\n-- failed with --\n{err}")
    return "\n\n".join(blocks)


def render_training_format_trajectory(
    task: str,
    current_code: str = "",
    feedback: str = "",
    attempts: list[dict[str, Any]] | None = None,
) -> str:
    """Render the episode the hypernetwork conditions on.

    The hypernet was distilled on ``## Task / ## Current Code / ## Review
    Feedback`` (the human-facing prompt template stays separate; the
    ``## Revision`` block is the generation target, not conditioning). To make
    the adapter a memory substrate, the conditioning now also carries **what has
    already been tried** (R2, #52): a ``## Previous Attempts`` section with the
    prior failing attempts and their errors. It is appended ONLY when there is
    history, so attempt-1 stays byte-identical to the distillation surface; the
    section is the new training surface for the RL stage.
    """
    out = (
        f"## Task\n{task}\n\n"
        f"## Current Code\n{current_code}\n\n"
        f"## Review Feedback\n{feedback}"
    )
    prior = _format_attempts(attempts)
    if prior:
        out += f"\n\n## Previous Attempts\n{prior}"
    return out


def _is_zeroshot_attempt(
    prompt_mode: str, action: Action, code_results: Mapping[str, str]
) -> bool:
    """The escalation floor's zero-shot base candidate: the FIRST code attempt for
    a subtask in ``escalate`` mode (no prior code). It runs the base model on a
    clean single-shot prompt (== the capability ceiling); repairs/re-codes escalate
    to the adapter, and keep-best floors the engine at base (issue #52)."""
    return (
        prompt_mode == "escalate"
        and action.name == "code"
        and (action.target_subtask or "") not in code_results
    )


def _effective_scaling(
    prompt_mode: str,
    action: Action,
    code_results: Mapping[str, str],
    base_scaling: float,
) -> float:
    """Adapter scaling for this attempt: 0 (base) for the zero-shot candidate, the
    configured scaling otherwise. Modes other than ``escalate`` are unaffected."""
    if _is_zeroshot_attempt(prompt_mode, action, code_results):
        return 0.0
    return base_scaling


def render_episode_adapter(
    action_name: str, target: str | None, state: Mapping[str, Any]
) -> str:
    """Episodic adapter conditioning: the RIGHT context for the current step.

    Keeps c3's trained ``## Task / ## Current Code / ## Review Feedback`` surface,
    but fills it with context-appropriate, FOCUSED content instead of the full spec
    at every step (#52 episodic design):

    - ``decompose``: the full task (it must see everything to split it).
    - ``code``/``repair``/``plan`` for a subtask: condensed overall goal + THIS
      sub-goal (description + acceptance check) + the subtask's current code + error
      — the local episode, so the model is focused on the immediate step.
    - ``integrate``: overall goal + ALL subtasks' accepted code + integration error.
    """
    overall = str(state.get("overall_goal", "") or "")
    subtasks = state.get("subtasks", [])
    by_name = {s.name: s for s in subtasks}
    entry = str(state.get("entry_point", "") or "the function")

    if action_name == "decompose":
        return render_training_format_trajectory(task=str(state.get("task", "")))

    if action_name == "integrate" or not target:
        best_code = state.get("best_code", {})
        code_results = state.get("code_results", {})
        parts = [
            f"# {s.name} (builds {s.builds or entry})\n"
            f"{best_code.get(s.name) or code_results.get(s.name, '')}"
            for s in subtasks
            if best_code.get(s.name) or code_results.get(s.name)
        ]
        int_fb = state.get("integration_feedback")
        sig = str(state.get("signature", "") or "").strip()
        sig_hint = f"\nStarter signature:\n{sig}" if sig else ""
        public = str(state.get("public_checks", "") or "").strip()
        public_hint = f"\nPublic checks:\n{public}" if public else ""
        task = (
            f"{overall}\n\nIntegrate the completed subtasks into `{entry}`."
            f"{sig_hint}{public_hint}"
        )
        return render_training_format_trajectory(
            task=task,
            current_code="\n\n".join(parts),
            feedback=int_fb.stderr if int_fb else "",
        )

    sub = by_name.get(target)
    if sub is None:
        return render_training_format_trajectory(
            task=overall or str(state.get("task", ""))
        )
    # SPECIFIC, function-named headers (not generic "## Task / ## Review
    # Feedback"): the hypernetwork bakes this episode into the adapter WEIGHTS, so
    # to the model it is learned knowledge, not text in its context. Distinctive
    # per-function headers give the recall-phrased prompt ("recall what you
    # learned about `name`...") a sharp anchor to cue (issue #52 (1)).
    name = sub.name
    goal = overall
    if sub.description:
        goal = f"{goal}\n{sub.description}" if goal else sub.description
    if sub.acceptance_check:
        goal += f"\nAcceptance: {sub.acceptance_check}"
    # Seed the FIRST code attempt with the real bare signature so the adapter
    # conveys the call contract (R2); once code exists, condition on that instead.
    entry_pt = str(state.get("entry_point", "") or "")
    current_code = state.get("best_code", {}).get(target) or state.get(
        "code_results", {}
    ).get(target, "")
    if not current_code and name == entry_pt:
        current_code = _bare_signature_stub(
            name, str(state.get("signature", "") or ""), str(state.get("task", ""))
        )
    # What went wrong AND the diagnosis, co-located, named for THIS function.
    fb = state.get("feedback", {}).get(target)
    err = fb.stderr if (fb is not None and fb.exit_code != 0) else ""
    diag = state.get("diagnosis", {}).get(target, "")
    if diag:
        err = f"{err}\nDiagnosis: {diag}".strip() if err else f"Diagnosis: {diag}"
    tried: list[str] = []
    for rec in state.get("trajectory", []):
        if rec.target_subtask != target:
            continue
        if rec.feedback and rec.feedback.exit_code != 0:
            snippet = rec.feedback.stderr.splitlines()[-1][:120]
            tried.append(f"- step {rec.step} ({rec.action_name}): {snippet}")
    delivery = ""
    if name == entry_pt and entry_pt:
        delivery = format_delivery_contract(
            entry_point=entry_pt,
            bare_signature=_bare_signature_stub(
                entry_pt,
                str(state.get("signature", "") or ""),
                str(state.get("task", "") or ""),
            ),
            public_checks=str(state.get("public_checks", "") or ""),
        )
    mission = f"## Mission `{name}`\n{goal}"
    if delivery:
        mission = f"{mission}\n\n## Required deliverable\n{delivery}"
    sections = [mission]
    if current_code:
        sections.append(f"## `{name}` — your last attempt\n{current_code}")
    if err:
        sections.append(f"## `{name}` — what you learned was wrong with it\n{err}")
    if tried:
        block = (
            "## approaches already tried (all failed public oracle)\n"
            "Do NOT retry these. Try a structurally different algorithm.\n"
            + "\n".join(tried[-3:])
        )
        sections.append(block)
    return "\n\n".join(sections)


def _split_spec(spec: str) -> tuple[str, str]:
    """Split an MBPP-style spec into (prose, doctest asserts), dropping ``\"\"\"``."""
    prose: list[str] = []
    asserts: list[str] = []
    seen_assert = False
    for line in spec.splitlines():
        if line.strip() in ('"""', "'''"):
            continue
        if line.lstrip().startswith(">>>") or seen_assert:
            seen_assert = True
            asserts.append(line)
        else:
            prose.append(line)
    return "\n".join(prose).strip(), "\n".join(asserts).strip()


def _bare_signature_stub(entry_point: str, signature: str, spec: str) -> str:
    """Bare ``def name(params):`` stub for episodic conditioning.

    LCB (and `rune run`) ship a ``class Solution: def name(self, a: T) -> R:``
    starter, but the engine emits a TOP-LEVEL function, so the episodic adapter
    must carry the bare signature (params minus ``self``) — issue #52 R2: the
    per-subtask context had dropped it, so the model invented parameter names
    (e.g. wrote ``(n, s)`` for a ``(s, k)`` contract, ignoring an argument).
    Falls back to doctest-derived arity when there is no usable starter.
    """
    src = signature.strip()
    if src:
        parse_src = f"{src} pass" if src.endswith(":") else src
        try:
            tree: ast.Module | None = ast.parse(parse_src)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not entry_point or node.name == entry_point
                ):
                    # Drop the `self`/`cls` receiver at the AST level (not by
                    # string surgery on the unparsed args).
                    node.args.args = [
                        a for a in node.args.args if a.arg not in ("self", "cls")
                    ]
                    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                    return f"def {node.name}({ast.unparse(node.args)}){ret}:"
    return _derive_signature(entry_point, spec)


def _derive_signature(entry_point: str, spec: str) -> str:
    """``def entry_point(arg1, ...):`` with arity inferred from the doctest call."""
    arity: int | None = None
    for line in spec.splitlines():
        if entry_point + "(" not in line:
            continue
        code = line.split(">>>", 1)[-1].strip()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == entry_point
            ):
                arity = len(node.args)
                break
        if arity is not None:
            break
    args = ", ".join(f"arg{i + 1}" for i in range(arity)) if arity else "*args"
    return f"def {entry_point}({args}):"


def render_reference_adapter(
    mode: str,
    spec: str,
    entry_point: str,
    *,
    signature: str = "",
    current_code: str = "",
    feedback: str = "",
) -> str:
    """Spec-in-adapter conditioning for the reference prompt modes (#52).

    The spec lives ONLY here (the prompt refers to the mission by name); empty
    sections are omitted. ``reference_a`` keeps c3's training surface (plain
    ``## Task``); ``reference_b`` uses Mission / Specification / Definition of
    Done + a ``## Current Code`` signature stub (Phase-1: condition on
    task + partial code).
    """
    if mode == "training_exact":
        # the byte-exact distillation surface: ## Task + EMPTY ## Current Code +
        # EMPTY ## Review Feedback (headers kept). Most faithful to c3's training.
        return render_training_format_trajectory(spec, current_code, feedback)

    sections: list[tuple[str, str]] = []
    if mode == "reference_b":
        prose, asserts = _split_spec(spec)
        sections.append(("Mission", entry_point))
        sections.append(("Specification", prose))
        sections.append(("Definition of Done", asserts))
        body = current_code or _derive_signature(entry_point, spec)
        sections.append(("Current Code", body))
        sections.append(("Review Feedback", feedback))
    elif mode == "reference_b1":
        # reference_b's richer content on OUR training headers (closer to
        # distribution): Mission under ## Task; signature + a comment under
        # ## Current Code; the doctest as "to be done" under ## Review Feedback.
        prose, asserts = _split_spec(spec)
        sig = signature or _derive_signature(entry_point, spec)
        to_do = asserts.replace(">>> ", "").strip()
        code = current_code or f"{sig}\n    # complete the implementation"
        sections.append(("Task", f"Mission: {entry_point}\n{prose}"))
        sections.append(("Current Code", code))
        sections.append(("Review Feedback", f"To be done: {to_do}" if to_do else ""))
    elif mode == "reference_c":
        # Strengthen the prompt<->context link: name the Mission explicitly (the
        # prompt refers to it by name), keep c3's ## Task surface, and anchor
        # ## Current Code with the REAL signature (real arg names bind better
        # than generic ones).
        sig = signature or _derive_signature(entry_point, spec)
        body = current_code or sig
        sections.append(("Mission", entry_point))
        sections.append(("Task", spec))
        sections.append(("Current Code", body))
        sections.append(("Review Feedback", feedback))
    else:  # reference_a
        sections.append(("Task", spec))
        sections.append(("Current Code", current_code))
        sections.append(("Review Feedback", feedback))
    return "\n\n".join(f"## {h}\n{c}" for h, c in sections if c.strip())


def state_to_ctx(state: RunState, action: Action | None = None) -> dict[str, Any]:
    subtasks = state["subtasks"]
    plans = state.get("plans", {})
    code_results = state.get("code_results", {})
    feedback_map = state.get("feedback", {})
    task = state["task"]

    ctx: dict[str, Any] = {
        "project": task[:_PROJECT_CAP],
        "task_description": task[:_PROJECT_CAP],
        "project_label": task[:_PROJECT_LABEL_CAP],
        "subtask_count": len(subtasks),
        # Required function name (benchmark tasks); "" for free-form `rune run`.
        # Named in the code/repair prompts so the model doesn't invent a name.
        "entry_point": state.get("entry_point", ""),
        "signature": state.get("signature", ""),
    }

    if action and action.name in _TARGETED_ACTIONS and not action.target_subtask:
        raise ValueError(
            f"Action {action.name!r} requires target_subtask but got "
            f"{action.target_subtask!r}"
        )

    if action and action.target_subtask:
        target_name = action.target_subtask
        subtask_idx, subtask_obj = next(
            ((i, s) for i, s in enumerate(subtasks) if s.name == target_name),
            (-1, None),
        )
        if subtask_obj is None:
            raise ValueError(f"Action targets unknown subtask {target_name!r}")
        ctx["subtask"] = subtask_obj
        ctx["subtask_name"] = target_name
        ctx["subtask_index"] = subtask_idx + 1
        ctx["total_subtasks"] = len(subtasks)
        ctx["plan"] = plans.get(target_name, "")
        ctx["target_subtask"] = target_name

        ctx["existing_code"] = code_results.get(target_name, "")

        subtask_fb = feedback_map.get(target_name)
        ctx["error_summary"] = subtask_fb.stderr[:2000] if subtask_fb else ""
        ctx["fix_guidance"] = state.get("diagnosis", {}).get(target_name, "")
        ctx["repair_brief"] = state.get("repair_briefs", {}).get(target_name, "")
        ctx["plan_rejection"] = state.get("plan_rejections", {}).get(target_name, "")
        entry_pt = str(state.get("entry_point", "") or target_name)
        ctx["bare_signature"] = _bare_signature_stub(
            entry_pt,
            str(state.get("signature", "") or ""),
            str(state.get("task", "") or ""),
        )
        ctx["delivery_contract"] = format_delivery_contract(
            entry_point=entry_pt,
            bare_signature=ctx["bare_signature"],
            public_checks=str(state.get("public_checks", "") or ""),
        )

        repair_history: list[str] = []
        code_trajectory: list[dict[str, Any]] = []
        for rec in state.get("trajectory", []):
            if rec.target_subtask != target_name:
                continue
            if rec.feedback and rec.feedback.exit_code != 0:
                repair_history.append(rec.feedback.stderr[:200])
            if rec.generated_code:
                code_trajectory.append(
                    {
                        "step": rec.step,
                        "action": rec.action_name,
                        "code": rec.generated_code[:_CODE_HISTORY_CAP],
                        "error": (
                            rec.feedback.stderr[:300]
                            if rec.feedback and rec.feedback.exit_code != 0
                            else ""
                        ),
                        "passed": bool(rec.feedback and rec.feedback.exit_code == 0),
                    }
                )
        ctx["repair_history"] = repair_history[-2:]
        ctx["code_trajectory"] = code_trajectory
        ctx["tried_and_failed"] = _format_tried_and_failed(code_trajectory)
        brief_text = ctx["repair_brief"]
        ctx["preserve_logic"] = bool(
            brief_text
            and any(
                f"failure_class: {fc}" in brief_text
                for fc in ("signature", "arity", "import")
            )
        )
    else:
        ctx["subtask"] = None
        ctx["target_subtask"] = None
        ctx["error_summary"] = ""
        ctx["fix_guidance"] = ""
        ctx["repair_history"] = []
        ctx["code_trajectory"] = []
        ctx["tried_and_failed"] = ""
        ctx["preserve_logic"] = False
        ctx["bare_signature"] = ""
        ctx["delivery_contract"] = ""

    ctx["integration_doc"] = "\n".join(
        f"- {s.name}: {s.description[:_INTEGRATION_DOC_LINE_CAP]}" for s in subtasks
    )
    ctx["code_outputs"] = code_results
    int_fb = state.get("integration_feedback")
    ctx["integration_error"] = int_fb.stderr if int_fb else ""

    return ctx


_JUDGE_SYSTEM = "You are a meticulous code reviewer hunting for edge-case bugs."
_COMPLEXITY_JUDGE_SYSTEM = (
    "You are a competitive-programming complexity analyst. "
    "Assess asymptotic TIME complexity from code structure."
)


def render_complexity_assessment_adapter(
    state: Mapping[str, Any],
    code: str,
) -> str:
    """Episodic adapter trajectory for static time-complexity assessment."""
    spec = str(state.get("task", "") or "")
    entry_point = str(state.get("entry_point", "") or "")
    signature = str(state.get("signature", "") or "")
    task = build_complexity_assessment_task(
        spec, entry_point, signature=signature
    )
    signals = static_complexity_signals(code)
    feedback = (
        "Public examples pass. Assess whether this implementation is fast enough "
        "for the stated Constraints at scale.\n\n"
        "Static signals:\n"
        + "\n".join(f"- {s}" for s in signals)
    )
    return render_training_format_trajectory(
        task=task,
        current_code=code,
        feedback=feedback,
    )


async def _run_complexity_judge(
    model: Any,
    state: Mapping[str, Any],
    code: str,
    run_config: dict[str, Any],
) -> ComplexityJudgeResult | None:
    """Adapter-backed complexity verdict when empirical big_o exceeds budget."""
    spec = str(state.get("task", "") or "")
    entry_point = str(state.get("entry_point", "") or "")
    constraints = parse_task_constraints(spec)
    if constraints is None:
        return None
    max_n = constraint_max_n(constraints)
    required_label, _ = allowed_complexity_for_max_n(max_n)
    static_signals = "\n".join(
        f"- {s}" for s in static_complexity_signals(code)
    ) or "- (none)"
    prompt = render_template(
        "prompt_complexity_judge",
        entry_point=entry_point,
        constraints_block=extract_constraints_block(spec),
        required_complexity=required_label,
        max_n=max_n,
        complexity_rubric=COMPLEXITY_ANALYSIS_RUBRIC,
        static_signals=static_signals,
        candidate_code=code,
    )
    try:
        result = await model.generate(
            prompt=prompt,
            system_prompt=_COMPLEXITY_JUDGE_SYSTEM,
            output_schema=ComplexityJudgeResult,
            max_tokens=run_config.get("complexity_judge_max_tokens", 384),
            temperature=run_config.get("complexity_judge_temperature", 0.1),
            thinking_budget=0,
        )
        return ComplexityJudgeResult.model_validate_json(result.text)
    except Exception:
        logger.warning(
            "complexity judge failed to produce a verdict; treating as sufficient"
        )
        return None


async def _run_constraint_complexity_oracle(
    model: Any,
    state: Mapping[str, Any],
    code: str,
    run_config: dict[str, Any],
) -> ScaleProbeOutcome | None:
    """Empirical big_o within budget; adapter judge fallback on timeout."""
    spec = str(state.get("task", "") or "")
    entry_point = str(state.get("entry_point", "") or "")
    public_checks = str(state.get("public_checks", "") or "")
    signature = str(state.get("signature", "") or "")
    if not constraint_scale_required(
        public_checks, entry_point, spec, signature=signature
    ):
        return None

    probe_config = ComplexityProbeConfig.from_state(state)
    timeout_s = float(run_config.get("complexity_empirical_timeout_s", 15.0))
    # Guarded: the empirical probe runs in a hard-killable subprocess so a slow
    # implementation can't stall the run (a thread can't be killed). Returns None
    # when the wall budget was exceeded -> escalate to the adapter judge.
    outcome = await asyncio.to_thread(
        check_constraint_scale_guarded,
        code,
        entry_point=entry_point,
        spec=spec,
        public_checks=public_checks,
        signature=signature,
        probe_config=probe_config,
        wall_timeout_s=timeout_s,
    )
    if outcome is not None:
        return outcome
    logger.info(
        "empirical complexity exceeded %.1fs budget; trying adapter judge",
        timeout_s,
    )

    if not run_config.get("complexity_judge_enabled", True):
        return ScaleProbeOutcome(required=True, ok=True)

    traj = render_complexity_assessment_adapter(state, code)
    scaling = float(run_config.get("adapter_scaling", 1.0))
    apply_episodic_adapter(model, traj, scaling=scaling)
    verdict = await _run_complexity_judge(model, state, code, run_config)
    if verdict is None:
        return ScaleProbeOutcome(required=True, ok=True)

    constraints = parse_task_constraints(spec)
    if constraints is None:
        return ScaleProbeOutcome(required=True, ok=True)
    max_n = constraint_max_n(constraints)
    allowed_label, _ = allowed_complexity_for_max_n(max_n)
    if verdict.sufficient:
        return ScaleProbeOutcome(required=True, ok=True)
    reason = verdict.reason.strip()
    suffix = f" {reason}" if reason else ""
    return ScaleProbeOutcome(
        required=True,
        ok=False,
        message=(
            f"constraint_scale: assessed {verdict.measured_complexity} "
            f"(adapter analysis); Constraints allow n≤{max_n} "
            f"— need {allowed_label} or better.{suffix}"
        ),
    )


async def _run_model_judge(
    model: Any,
    spec: str,
    entry_point: str,
    code: str,
    run_config: dict[str, Any],
) -> JudgeResult | None:
    """Ask the model for a grounded correctness verdict on *code*.

    Returns the parsed ``JudgeResult`` or ``None`` if generation/parse fails
    (fail-open: a flaky judge must not block already-passing code). Runs with the
    adapter currently loaded from the code step (the adapted agent judging its own
    work).
    """
    prompt = render_template(
        "prompt_judge",
        entry_point=entry_point,
        task_description=spec,
        candidate_code=code,
    )
    try:
        result = await model.generate(
            prompt=prompt,
            system_prompt=_JUDGE_SYSTEM,
            output_schema=JudgeResult,
            max_tokens=run_config.get("judge_max_tokens", 256),
            temperature=run_config.get("judge_temperature", 0.2),
            thinking_budget=0,
        )
        return JudgeResult.model_validate_json(result.text)
    except Exception:
        logger.warning("model judge failed to produce a verdict; treating as correct")
        return None


async def step_node(state: RunState, config: RunnableConfig) -> dict[str, Any]:
    configurable: dict[str, Any] = config.get("configurable", {})
    model = configurable["model"]
    run_config: dict[str, Any] = configurable.get("run_config", {})

    # `decompose` always runs first and the MODEL decides 1 vs N subtasks (the
    # decompose prompt instructs ONE subtask for a self-contained function). No
    # fragile phrase/word-count heuristic pre-empts that decision.
    actions = select_action(dict(state))
    if not actions:
        return {"actions": [], "budget_remaining": state["budget_remaining"]}

    temperature = run_config.get("temperature", 0.3)
    adapter_scaling = run_config.get("adapter_scaling", 1.0)
    repetition_penalty = run_config.get("repetition_penalty", 1.1)
    top_p = run_config.get("top_p", 0.9)
    presence_penalty = run_config.get("presence_penalty", 0.0)
    thinking_budget = run_config.get("thinking_budget", 1024)

    results: list[tuple[Action, str, str, str | None, str, str, str]] = []
    for action in actions:
        # Episodic invariant: each action gets a fresh hypernet adapter via
        # apply_episodic_adapter() immediately before inference — never reuse
        # a prior step's LoRA weights (see tests/unit/test_adapter_episodic_swap.py).
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()

        ctx = state_to_ctx(state, action)
        feedback_text = ctx.get("fix_guidance") or ctx.get("error_summary") or ""
        prompt_mode = run_config.get("prompt_mode", "full")
        _ref_modes = (
            "training_exact",
            "reference_a",
            "reference_b",
            "reference_b1",
            "reference_c",
        )
        if _is_zeroshot_attempt(prompt_mode, action, state.get("code_results", {})):
            # Escalation floor (#52): the zero-shot base candidate uses the CLEAN
            # single-shot prompt (== capability ceiling), NOT the plan/subtask
            # framing — so the floor candidate matches base instead of being
            # degraded by "follow the architecture plan" contamination. Adapter is
            # off (scaling 0), so the trajectory is immaterial.
            trajectory_text = render_training_format_trajectory(
                task=ctx["task_description"]
            )
            prompt_text = render_template("prompt_zeroshot", **ctx)
        elif prompt_mode in ("episodic", "escalate"):
            # Episodic recall format (#52): adapter carries the right context per
            # step. In `escalate` this is the adapter-on repair/re-code path (the
            # zero-shot base already returned above) — so the winning architecture
            # conditions and trains on the new recall format.
            trajectory_text = render_episode_adapter(
                action.name, action.target_subtask, state
            )
            prompt_text = render_template(
                _EPISODIC_PROMPTS.get(action.name, action.prompt_template), **ctx
            )
        elif prompt_mode in _ref_modes and action.name in ("code", "repair"):
            # spec-in-adapter (#52): the spec lives only in the conditioning; the
            # prompt refers to the mission by name.
            trajectory_text = render_reference_adapter(
                prompt_mode,
                ctx["task_description"],
                ctx.get("entry_point", ""),
                signature=ctx.get("signature", ""),
                current_code=ctx.get("existing_code", ""),
                feedback=feedback_text,
            )
            prompt_text = render_template(f"prompt_{prompt_mode}", **ctx)
        else:
            # Prior attempts (all code/repair tries before the current one, which
            # is in ## Current Code) become the episode the adapter carries (R2).
            prior_attempts = ctx.get("code_trajectory", [])[:-1]
            trajectory_text = render_training_format_trajectory(
                task=ctx["task_description"],
                current_code=ctx.get("existing_code", ""),
                feedback=feedback_text,
                attempts=prior_attempts,
            )
            prompt_text = render_template(action.prompt_template, **ctx)

        eff_scaling = _effective_scaling(
            prompt_mode, action, state.get("code_results", {}), adapter_scaling
        )
        adapter_id = apply_episodic_adapter(model, trajectory_text, scaling=eff_scaling)
        result = await model.generate(
            prompt=prompt_text,
            system_prompt=action.system_prompt,
            output_schema=action.output_schema,
            max_tokens=run_config.get("max_tokens", 2048),
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=run_config.get("no_repeat_ngram_size", 0),
            presence_penalty=presence_penalty,
            thinking_budget=thinking_budget,
        )
        raw_text = result.text
        needs_continuation = result.truncated
        if action.name in ("code", "repair", "integrate") and needs_continuation:
            cont_multiplier = run_config.get("cont_multiplier", 1.53)
            cont_no_repeat = run_config.get("no_repeat_ngram_size", 12)
            cont_scaling = adapter_scaling * cont_multiplier
            accumulated_code = extract_partial_code(result.text)
            cont_budget = run_config.get("cont_budget", 5)
            cont_round = 0
            empty_rounds = 0

            cont_sys = CONT_SYSTEM_PROMPT
            cont_user = render_template("prompt_code_continue", **ctx)

            while cont_budget > 0 and empty_rounds < 2:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
                cont_round += 1

                cont_traj = render_training_format_trajectory(
                    task=ctx["task_description"],
                    current_code=accumulated_code[-_ACCUMULATED_CODE_CAP:],
                    feedback=ctx.get("fix_guidance") or ctx.get("error_summary") or "",
                )

                apply_episodic_adapter(model, cont_traj, scaling=cont_scaling)

                result = await model.generate_continuation(
                    system_prompt=cont_sys,
                    user_prompt=cont_user,
                    assistant_prefix=accumulated_code,
                    max_tokens=run_config.get("max_tokens", 2048),
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    no_repeat_ngram_size=cont_no_repeat,
                    presence_penalty=presence_penalty,
                )

                new_chunk = result.text
                degen = degeneration_score(new_chunk)
                logger.info(
                    "continuation round %d: +%d chars, degen=%.2f, truncated=%s",
                    cont_round,
                    len(new_chunk),
                    degen,
                    result.truncated,
                )

                if degen > 0.5:
                    logger.warning(
                        "Degeneration detected (%.2f), stopping continuation",
                        degen,
                    )
                    break

                if new_chunk.strip():
                    accumulated_code += new_chunk
                    empty_rounds = 0
                else:
                    empty_rounds += 1

                cont_budget -= 1

                if validate_syntax(accumulated_code):
                    logger.info("Accumulated code validates — exiting continuation")
                    break

                if not result.truncated:
                    break

            logger.info(
                "continuation done: %d rounds, %d chars, syntax_valid=%s",
                cont_round,
                len(accumulated_code),
                validate_syntax(accumulated_code),
            )
            raw_text = accumulated_code
        elif result.truncated:
            logger.warning(
                "Truncated output for %s/%s after completion retry",
                action.name,
                action.target_subtask or "",
            )

        target_name = action.target_subtask or ""
        output_text = (
            extract_partial_code(raw_text) if action.executes_code else raw_text
        )
        results.append(
            (
                action,
                target_name,
                raw_text,
                adapter_id,
                trajectory_text,
                prompt_text,
                output_text,
            )
        )

        if mlflow.active_run() is not None:
            prefix = f"step_{state['step']}/{action.name}"
            if action.target_subtask:
                prefix += f"_{action.target_subtask}"
            mlflow.log_text(trajectory_text, f"{prefix}/trajectory.txt")
            mlflow.log_text(prompt_text, f"{prefix}/prompt.txt")
            mlflow.log_text(result.text, f"{prefix}/output.txt")
            # GOAL-3 pre-reg (g): adapter-conditioning vs prompt token budget per
            # turn — the adapter-as-memory thesis instrument (prompt ~flat, adapter
            # trajectory grows across repair/continuation). Logged at the engine
            # step, not a wrapper, so it reflects exactly what conditioned the
            # adapter vs what the model read in the prompt.
            mlflow.log_metric(
                "adapter_cond_tokens",
                model.count_tokens(trajectory_text),
                step=state["step"],
            )
            mlflow.log_metric(
                "prompt_tokens",
                model.count_tokens(prompt_text),
                step=state["step"],
            )

    # extract_partial_code is the single code-extraction primitive (de-fence the
    # freeform model output). Computed once here and threaded into parse_output
    # below so the code recorded in state is exactly the code executed in the
    # sandbox.
    code_map: dict[str, str] = {}
    code_action_names: list[str] = []
    for a, name, text, _, _traj, _prompt, _out in results:
        if a.executes_code:
            code_map[name] = extract_partial_code(text)
            code_action_names.append(name)

    # In-loop correctness signal: append the spec's PUBLIC doctest examples to the
    # candidate so a wrong/crashing impl fails the sandbox and routes to repair
    # (a bare def otherwise exits 0 — module-load only). Held-out task tests are
    # never used here; pass@1 still gates on the full held-out set at scoring.
    spec = state.get("task", "")
    entry_point = state.get("entry_point", "")
    # Episodic design: a subtask's own model-authored acceptance_check is its
    # in-loop signal; fall back to the spec's public examples (whole-task / N=1 /
    # integrate) when the subtask has none.
    _subtask_check = {s.name: s.acceptance_check for s in state.get("subtasks", [])}
    probes: dict[str, tuple[str, bool, bool]] = {}
    for name in code_action_names:
        probes[name] = build_code_probe(name, code_map[name], state)
    sandbox_results = await asyncio.gather(
        *[
            asyncio.to_thread(run_in_sandbox, probes[name][0])
            for name in code_action_names
        ]
    )
    feedback_map = {
        name: apply_oracle_fail_closed(
            probes[name][1],
            probes[name][2],
            Feedback(stdout=fb.stdout, stderr=fb.stderr, exit_code=fb.exit_code),
        )
        for name, fb in zip(code_action_names, sandbox_results, strict=True)
    }
    for name in code_action_names:
        _fired = probes[name][1]
        # integrate's target is "" -> a trailing-slash metric name MLflow rejects;
        # label it explicitly.
        _label = name or "integrate"
        _fb = feedback_map[name]
        _resolved_check = resolve_in_loop_check(
            name, _subtask_check.get(name, ""), state
        )
        _n_checks = (
            len(split_acceptance_checks(_resolved_check))
            if _resolved_check.strip()
            else 0
        )
        logger.info(
            "oracle for %s: %s (%d check(s))",
            _label,
            "fired (public examples)" if _fired else "fallback (module-load only)",
            _n_checks,
        )
        if _fired and _fb.exit_code != 0:
            _detail = (_fb.stderr or _fb.stdout or "").strip()[:500]
            logger.info("oracle check failed for %s: %s", _label, _detail)
        if mlflow.active_run() is not None:
            mlflow.log_metric(f"oracle_fired/{_label}", int(_fired), step=state["step"])
            if _n_checks:
                mlflow.log_metric(
                    f"oracle_n_checks/{_label}", _n_checks, step=state["step"]
                )

    # Model-judge (in-loop, always on): the public example is necessary but not
    # sufficient — code can pass its one public case yet be wrong on a held-out
    # input (e.g. integer-division edge cases). For each unit that passed the
    # sandbox, ask the model for a SPECIFIC failing input; a grounded verdict
    # flips feedback to failure so the existing diagnose->repair routing engages,
    # and the named input becomes the repair signal carried in the adapter.
    if run_config.get("model_judge", True):
        for name in code_action_names:
            if feedback_map[name].exit_code != 0:
                continue  # already failing — nothing for the judge to add
            verdict = await _run_model_judge(
                model, spec, entry_point, code_map[name], run_config
            )
            if verdict is not None and not verdict.correct and verdict.failing_input:
                feedback_map[name] = Feedback(
                    stdout="",
                    stderr=(
                        f"Correctness judge: wrong on input {verdict.failing_input}. "
                        f"{verdict.reason}"
                    ),
                    exit_code=1,
                )
                logger.info(
                    "judge flipped %s to failing: %s",
                    name or "integrate",
                    verdict.reason,
                )
                if mlflow.active_run() is not None:
                    mlflow.log_metric(
                        f"judge_flagged/{name or 'integrate'}", 1, step=state["step"]
                    )

    # Constraint-scale oracle: empirical big_o when it finishes within budget;
    # otherwise hotswap the complexity-assessment adapter for a static verdict.
    for name in code_action_names:
        if feedback_map[name].exit_code != 0:
            continue
        cx_outcome = await _run_constraint_complexity_oracle(
            model, state, code_map[name], run_config
        )
        if cx_outcome is not None and cx_outcome.required and not cx_outcome.ok:
            feedback_map[name] = Feedback(
                stdout="",
                stderr=format_requirements_feedback((cx_outcome.message,)),
                exit_code=1,
            )
            logger.info(
                "constraint complexity failed for %s: %s",
                name or "integrate",
                cx_outcome.message,
            )
            if mlflow.active_run() is not None:
                mlflow.log_metric(
                    f"complexity_failed/{name or 'integrate'}",
                    1,
                    step=state["step"],
                )

    # Task requirements oracle (benchmark/LCB): pluggable checks that activate
    # only from structured task evidence (starter, public_checks, Constraints).
    if str(state.get("public_checks", "") or "").strip():
        for name in code_action_names:
            if feedback_map[name].exit_code != 0:
                continue
            ok, deficiencies = evaluate_state_requirements(
                state,
                code_map[name],
                skip_kinds=frozenset({"constraint_scale"}),
            )
            if not ok:
                msg = format_requirements_feedback(deficiencies)
                feedback_map[name] = Feedback(stdout="", stderr=msg, exit_code=1)
                logger.info(
                    "task requirements failed for %s: %s",
                    name or "integrate",
                    msg,
                )
                if mlflow.active_run() is not None:
                    mlflow.log_metric(
                        f"requirements_failed/{name or 'integrate'}",
                        1,
                        step=state["step"],
                    )

    brief_updates: dict[str, Any] = {}
    if state.get("repair_brief_enabled", True):
        from rune.engine.repair_brief import build_repair_brief  # noqa: PLC0415

        briefs = dict(state.get("repair_briefs", {}))
        replan = dict(state.get("replan_targets", {}))
        plans = dict(state.get("plans", {}))
        for name in code_action_names:
            fb = feedback_map.get(name)
            if fb is None or fb.exit_code == 0:
                continue
            sub = next((s for s in state.get("subtasks", []) if s.name == name), None)
            brief = build_repair_brief(
                fb.stderr,
                entry_point=str(state.get("entry_point", "") or ""),
                signature=str(state.get("signature", "") or ""),
                plan=str(state.get("plans", {}).get(name, "") or ""),
                overall_goal=str(state.get("overall_goal", "") or ""),
                acceptance_check=sub.acceptance_check if sub else "",
                subtask_description=sub.description if sub else "",
                complexity_repair_preserve_logic=bool(
                    state.get("complexity_repair_preserve_logic", True)
                ),
            )
            if brief is None:
                continue
            briefs[name] = brief.format_block()
            if brief.replan_recommended and state.get("replan_on_complexity", True):
                replan[name] = True
                plans.pop(name, None)
                logger.info(
                    "replan recommended for %s (%s)",
                    name,
                    brief.failure_class,
                )
        brief_updates = {"repair_briefs": briefs, "replan_targets": replan}
        if plans != state.get("plans", {}):
            brief_updates["plans"] = plans

    # Thread an accumulating running state through siblings so each parse_output
    # builds its full maps from the prior sibling's applied change. Reusing a
    # frozen dict(state) snapshot per sibling let the last-merged sibling's stale
    # copy clobber earlier siblings' real updates (code_passed/retries/...).
    updates: dict[str, Any] = dict(brief_updates)
    running = dict(state)
    running.update(brief_updates)
    for action, target_name, raw, _, _traj, _prompt, _out in results:
        fb = feedback_map.get(target_name)
        partial = parse_output(action, raw, fb, running, code=code_map.get(target_name))
        updates.update(partial)
        running.update(partial)

    records = [
        StepRecord(
            step=state["step"],
            action_name=a.name,
            target_subtask=name,
            adapter_id=aid,
            feedback=feedback_map.get(name),
            generated_code=code_map.get(name) or None,
            trajectory_text=traj,
            prompt_text=prompt,
            output_text=out,
        )
        for a, name, _, aid, traj, prompt, out in results
    ]
    running_final = dict(state)
    running_final.update(updates)
    updates["actions"] = select_action(running_final)
    updates["current_adapter"] = results[-1][3] if results else state["current_adapter"]
    updates["trajectory"] = state["trajectory"] + records
    updates["step"] = state["step"] + 1
    updates["budget_remaining"] = state["budget_remaining"] - 1
    return updates


def should_continue(state: RunState) -> str:
    if state["budget_remaining"] <= 0:
        return "done"
    if not select_action(dict(state)):
        return "done"
    return "continue"


def create_engine() -> CompiledStateGraph:  # type: ignore[type-arg]
    graph = StateGraph(RunState)
    graph.add_node("step", step_node)
    graph.set_entry_point("step")
    graph.add_conditional_edges(
        "step",
        should_continue,
        {
            "continue": "step",
            "done": END,
        },
    )
    return graph.compile()
