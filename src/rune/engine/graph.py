"""Single-node LangGraph engine: step_node + should_continue loop."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import mlflow
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rune.engine.continuation import (
    dedup_code,
    degeneration_score,
    extract_partial_code,
    merge_overlap,
)
from rune.engine.parse import CodeResult, IntegrateResult, parse_output, render_template
from rune.engine.policy import select_action
from rune.engine.state import (
    _CODE_HISTORY_CAP,
    Action,
    Feedback,
    RunState,
    StepRecord,
    Subtask,
)
from rune.sandbox.executor import run_in_sandbox

logger = logging.getLogger(__name__)

_SIMPLE_SIGNALS = (
    "write a function",
    "implement a function",
    "implement a method",
    "create a function",
    "write a method",
    "create a method",
    "write a class",
    "implement a class",
    "create a class",
)

_SIMPLE_WORD_LIMIT = 200


def _is_simple_task(task: str) -> bool:
    """Heuristic: short task with a single-unit signal skips decomposition."""
    if len(task.split()) >= _SIMPLE_WORD_LIMIT:
        return False
    lower = task.lower()
    return any(sig in lower for sig in _SIMPLE_SIGNALS)


def state_to_ctx(state: RunState, action: Action | None = None) -> dict[str, Any]:
    subtasks = state["subtasks"]
    plans = state.get("plans", {})
    code_results = state.get("code_results", {})
    interfaces = state.get("interfaces", {})
    feedback_map = state.get("feedback", {})
    task = state["task"]

    ctx: dict[str, Any] = {
        "project": task,
        "task_description": task,
        "project_label": task[:200],
        "subtask_count": len(subtasks),
    }

    _TARGETED_ACTIONS = {"plan", "code", "repair"}
    if action and action.name in _TARGETED_ACTIONS and not action.target_subtask:
        raise ValueError(
            f"Action {action.name!r} requires target_subtask but got "
            f"{action.target_subtask!r}"
        )

    if action and action.target_subtask:
        target_name = action.target_subtask
        subtask_obj = next((s for s in subtasks if s.name == target_name), None)
        if subtask_obj is None:
            raise ValueError(f"Action targets unknown subtask {target_name!r}")
        subtask_idx = next(
            (i for i, s in enumerate(subtasks) if s.name == target_name), 0
        )
        ctx["subtask"] = subtask_obj
        ctx["subtask_name"] = target_name
        ctx["subtask_index"] = subtask_idx + 1
        ctx["total_subtasks"] = len(subtasks)
        ctx["plan"] = plans.get(target_name, "")
        ctx["target_subtask"] = target_name

        dep_ifaces: list[str] = []
        for dep in subtask_obj.depends_on:
            if dep in interfaces:
                dep_ifaces.append(f"# {dep}\n{interfaces[dep]}")
        ctx["dependency_interfaces"] = "\n".join(dep_ifaces)
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
                code_trajectory.append({
                    "step": rec.step,
                    "action": rec.action_name,
                    "code": rec.generated_code[:_CODE_HISTORY_CAP],
                    "error": (
                        rec.feedback.stderr[:300]
                        if rec.feedback and rec.feedback.exit_code != 0
                        else ""
                    ),
                    "passed": bool(
                        rec.feedback and rec.feedback.exit_code == 0
                    ),
                })
        ctx["repair_history"] = repair_history[-2:]
        ctx["code_trajectory"] = code_trajectory
    else:
        ctx["target_subtask"] = None
        ctx["error_summary"] = ""
        ctx["fix_guidance"] = ""
        ctx["repair_history"] = []
        ctx["code_trajectory"] = []

    ctx["integration_doc"] = "\n".join(f"- {s.name}: {s.description}" for s in subtasks)
    ctx["skeletons"] = code_results
    ctx["code_outputs"] = code_results
    int_fb = state.get("integration_feedback")
    ctx["integration_error"] = int_fb.stderr if int_fb else ""

    return ctx


async def step_node(state: RunState, config: RunnableConfig) -> dict[str, Any]:
    configurable: dict[str, Any] = config.get("configurable", {})
    model = configurable["model"]
    run_config: dict[str, Any] = configurable.get("run_config", {})

    # Complexity gate: simple tasks skip decomposition
    gate_fired = False
    if not state["subtasks"] and _is_simple_task(state["task"]):
        synthetic = Subtask(
            name="_main",
            description=state["task"],
            depends_on=[],
        )
        state = {**state, "subtasks": [synthetic], "plans": {"_main": state["task"]}}
        gate_fired = True

    actions = select_action(dict(state))
    if not actions:
        return {"actions": [], "budget_remaining": state["budget_remaining"]}

    temperature = run_config.get("temperature", 0.3)
    adapter_scaling = run_config.get("adapter_scaling", 1.0)
    repetition_penalty = run_config.get("repetition_penalty", 1.1)
    top_p = run_config.get("top_p", 0.9)

    results: list[tuple[Action, str, str, str | None]] = []
    cont_budget_spent = 0
    for action in actions:
        ctx = state_to_ctx(state, action)
        trajectory_text = render_template(action.trajectory_template, **ctx)
        prompt_text = render_template(action.prompt_template, **ctx)

        adapter = model.generate_adapter(trajectory_text)
        scaled_sd = {
            k: v * adapter_scaling if "lora_B" in k else v
            for k, v in adapter.state_dict.items()
        }
        model.hotswap_adapter(scaled_sd)
        result = await model.generate(
            prompt=prompt_text,
            system_prompt=action.system_prompt,
            output_schema=action.output_schema,
            max_tokens=run_config.get("max_tokens", 2048),
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
        )
        raw_text = result.text
        if action.name in ("code", "repair") and result.truncated:
            cont_multiplier = run_config.get("cont_multiplier", 1.53)
            cont_no_repeat = run_config.get("no_repeat_ngram_size", 12)
            cont_scaling = adapter_scaling * cont_multiplier
            accumulated_code = extract_partial_code(result.text)
            budget = state["budget_remaining"] - 1
            empty_rounds = 0

            while result.truncated and budget > 0 and empty_rounds < 2:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()

                cont_ctx = {
                    **ctx,
                    "accumulated_code": accumulated_code,
                    "resume_tail": "\n".join(accumulated_code.splitlines()[-4:]),
                }
                cont_traj = render_template("code_continue", **cont_ctx)
                cont_prompt = render_template("prompt_code_continue", **cont_ctx)

                cont_adapter = model.generate_adapter(cont_traj)
                cont_sd = {
                    k: v * cont_scaling if "lora_B" in k else v
                    for k, v in cont_adapter.state_dict.items()
                }
                model.hotswap_adapter(cont_sd)

                result = await model.generate(
                    prompt=cont_prompt,
                    system_prompt=action.system_prompt,
                    output_schema=CodeResult,
                    max_tokens=run_config.get("max_tokens", 2048),
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    no_repeat_ngram_size=cont_no_repeat,
                    thinking_budget=0,
                    skip_completion_retry=True,
                )

                chunk = extract_partial_code(result.text)
                chunk = merge_overlap(accumulated_code, chunk)
                chunk = dedup_code(chunk, accumulated_code)

                degen = degeneration_score(chunk)
                logger.info(
                    "continuation round: +%d chars, degen=%.2f",
                    len(chunk),
                    degen,
                )

                if chunk.strip():
                    accumulated_code = (
                        accumulated_code.rstrip()
                        + "\n"
                        + chunk.strip()
                        + "\n"
                    )
                    empty_rounds = 0
                else:
                    empty_rounds += 1

                budget -= 1
                cont_budget_spent += 1

            raw_text = json.dumps({"code": accumulated_code})
        elif result.truncated:
            logger.warning(
                "Truncated output for %s/%s after completion retry",
                action.name, action.target_subtask or "",
            )

        target_name = action.target_subtask or ""
        results.append((action, target_name, raw_text, adapter.adapter_id))

        if mlflow.active_run() is not None:
            prefix = f"step_{state['step']}/{action.name}"
            if action.target_subtask:
                prefix += f"_{action.target_subtask}"
            mlflow.log_text(trajectory_text, f"{prefix}/trajectory.txt")
            mlflow.log_text(prompt_text, f"{prefix}/prompt.txt")
            mlflow.log_text(result.text, f"{prefix}/output.txt")

    def _parse_action_code(action: Action, raw_json: str) -> str:
        if action.name == "integrate":
            return IntegrateResult.model_validate_json(raw_json).code
        return CodeResult.model_validate_json(raw_json).code

    code_map: dict[str, str] = {}
    code_action_names: list[str] = []
    for a, name, text, _ in results:
        if a.executes_code:
            code_map[name] = _parse_action_code(a, text)
            code_action_names.append(name)

    sandbox_results = await asyncio.gather(
        *[
            asyncio.to_thread(run_in_sandbox, code_map[name])
            for name in code_action_names
        ]
    )
    feedback_map = {
        name: Feedback(stdout=fb.stdout, stderr=fb.stderr, exit_code=fb.exit_code)
        for name, fb in zip(code_action_names, sandbox_results, strict=True)
    }

    updates: dict[str, Any] = {}
    for action, target_name, raw, _ in results:
        fb = feedback_map.get(target_name)
        partial = parse_output(action, raw, fb, dict(state))
        for k, v in partial.items():
            if isinstance(v, dict) and isinstance(updates.get(k), dict):
                updates[k] = {**updates[k], **v}
            else:
                updates[k] = v

    records = [
        StepRecord(
            step=state["step"],
            action_name=a.name,
            target_subtask=name,
            adapter_id=aid,
            feedback=feedback_map.get(name),
            generated_code=code_map.get(name, "")[:_CODE_HISTORY_CAP] or None,
        )
        for a, name, _, aid in results
    ]
    if gate_fired:
        updates.setdefault("subtasks", state["subtasks"])
        updates.setdefault("plans", state["plans"])
    updates["actions"] = actions
    updates["current_adapter"] = results[-1][3] if results else state["current_adapter"]
    updates["trajectory"] = state["trajectory"] + records
    updates["step"] = state["step"] + 1
    updates["budget_remaining"] = state["budget_remaining"] - 1 - cont_budget_spent
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
