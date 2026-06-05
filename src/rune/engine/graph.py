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

from rune.engine.continuation import (
    CONT_SYSTEM_PROMPT,
    degeneration_score,
    extract_partial_code,
    strip_self_tests,
    validate_syntax,
)
from rune.engine.oracle import build_probe
from rune.engine.parse import JudgeResult, parse_output, render_template
from rune.engine.policy import select_action
from rune.engine.state import (
    _CODE_HISTORY_CAP,
    Action,
    Feedback,
    RunState,
    StepRecord,
)
from rune.model.adapter import scale_lora_b
from rune.sandbox.executor import run_in_sandbox

logger = logging.getLogger(__name__)

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
        code_results = state.get("code_results", {})
        parts = [
            f"# {s.name} (builds {s.builds or entry})\n{code_results[s.name]}"
            for s in subtasks
            if code_results.get(s.name)
        ]
        int_fb = state.get("integration_feedback")
        task = f"{overall}\n\nIntegrate the completed subtasks into `{entry}`."
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
    task = f"{overall}\n\nSubtask `{sub.name}`: {sub.description}"
    if sub.acceptance_check:
        task += f"\nAcceptance: {sub.acceptance_check}"
    fb = state.get("feedback", {}).get(target)
    err = fb.stderr if (fb is not None and fb.exit_code != 0) else ""
    if not err:
        err = state.get("diagnosis", {}).get(target, "")
    return render_training_format_trajectory(
        task=task,
        current_code=state.get("code_results", {}).get(target, ""),
        feedback=err,
    )


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
        ctx["error_summary"] = subtask_fb.stderr[:500] if subtask_fb else ""
        ctx["fix_guidance"] = state.get("diagnosis", {}).get(target_name, "")

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
    else:
        ctx["subtask"] = None
        ctx["target_subtask"] = None
        ctx["error_summary"] = ""
        ctx["fix_guidance"] = ""
        ctx["repair_history"] = []
        ctx["code_trajectory"] = []

    ctx["integration_doc"] = "\n".join(
        f"- {s.name}: {s.description[:_INTEGRATION_DOC_LINE_CAP]}" for s in subtasks
    )
    ctx["code_outputs"] = code_results
    int_fb = state.get("integration_feedback")
    ctx["integration_error"] = int_fb.stderr if int_fb else ""

    return ctx


_JUDGE_SYSTEM = "You are a meticulous code reviewer hunting for edge-case bugs."


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
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()

        ctx = state_to_ctx(state, action)
        feedback_text = ctx.get("fix_guidance") or ctx.get("error_summary") or ""
        prompt_mode = run_config.get("prompt_mode", "full")
        _ref_modes = (
            "training_exact",
            "reference_a", "reference_b", "reference_b1", "reference_c",
        )
        if prompt_mode == "episodic":
            # Episodic design (#52): adapter carries the right context per step;
            # the prompt is a thin pointer to the immediate sub-goal (no spec leak).
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

        adapter = model.generate_adapter(trajectory_text)
        scaled_sd = scale_lora_b(adapter.state_dict, adapter_scaling)
        model.hotswap_adapter(scaled_sd)
        adapter_id = adapter.adapter_id
        del adapter, scaled_sd
        result = await model.generate(
            prompt=prompt_text,
            system_prompt=action.system_prompt,
            output_schema=action.output_schema,
            max_tokens=run_config.get("max_tokens", 2048),
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
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

                cont_adapter = model.generate_adapter(cont_traj)
                cont_sd = scale_lora_b(cont_adapter.state_dict, cont_scaling)
                model.hotswap_adapter(cont_sd)
                del cont_adapter, cont_sd

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
                "adapter_cond_tokens", model.count_tokens(trajectory_text),
                step=state["step"],
            )
            mlflow.log_metric(
                "prompt_tokens", model.count_tokens(prompt_text),
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
    probes: dict[str, tuple[str, bool]] = {
        name: build_probe(strip_self_tests(code_map[name]), spec, entry_point)
        for name in code_action_names
    }
    sandbox_results = await asyncio.gather(
        *[
            asyncio.to_thread(run_in_sandbox, probes[name][0])
            for name in code_action_names
        ]
    )
    feedback_map = {
        name: Feedback(stdout=fb.stdout, stderr=fb.stderr, exit_code=fb.exit_code)
        for name, fb in zip(code_action_names, sandbox_results, strict=True)
    }
    for name in code_action_names:
        _fired = probes[name][1]
        # integrate's target is "" -> a trailing-slash metric name MLflow rejects;
        # label it explicitly.
        _label = name or "integrate"
        logger.info(
            "oracle for %s: %s",
            _label,
            "fired (public examples)" if _fired else "fallback (module-load only)",
        )
        if mlflow.active_run() is not None:
            mlflow.log_metric(
                f"oracle_fired/{_label}", int(_fired), step=state["step"]
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

    # Thread an accumulating running state through siblings so each parse_output
    # builds its full maps from the prior sibling's applied change. Reusing a
    # frozen dict(state) snapshot per sibling let the last-merged sibling's stale
    # copy clobber earlier siblings' real updates (code_passed/retries/...).
    updates: dict[str, Any] = {}
    running = dict(state)
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
    updates["actions"] = actions
    updates["current_adapter"] = results[-1][3] if results else state["current_adapter"]
    updates["trajectory"] = state["trajectory"] + records
    updates["step"] = state["step"] + 1
    updates["budget_remaining"] = state["budget_remaining"] - 1
    return updates


def should_continue(state: RunState) -> str:
    if not state["actions"] or state["budget_remaining"] <= 0:
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
