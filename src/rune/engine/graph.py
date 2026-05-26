"""Single-node LangGraph engine: step_node + should_continue loop."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rune.engine.parse import parse_output, render_template
from rune.engine.policy import is_simple_task, select_action
from rune.engine.state import Action, Feedback, RunState, StepRecord, Subtask
from rune.sandbox.executor import run_in_sandbox


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
        ctx["error_summary"] = (
            subtask_fb.stderr[:500] if subtask_fb else ""
        )
        ctx["fix_guidance"] = (
            state.get("diagnosis", {}).get(target_name, "")
        )

        repair_history: list[str] = []
        for rec in state.get("trajectory", []):
            if (
                rec.target_subtask == target_name
                and rec.feedback
                and rec.feedback.exit_code != 0
            ):
                repair_history.append(rec.feedback.stderr[:200])
        ctx["repair_history"] = repair_history[-2:]
    else:
        ctx["target_subtask"] = None
        ctx["error_summary"] = ""
        ctx["fix_guidance"] = ""
        ctx["repair_history"] = []

    ctx["integration_doc"] = "\n".join(
        f"- {s.name}: {s.description}" for s in subtasks
    )
    ctx["skeletons"] = code_results
    ctx["code_outputs"] = code_results
    int_fb = state.get("integration_feedback")
    ctx["integration_error"] = int_fb.stderr if int_fb else ""

    return ctx


async def step_node(state: RunState, config: RunnableConfig) -> dict[str, Any]:
    configurable: dict[str, Any] = config.get("configurable", {})
    model = configurable["model"]
    run_config: dict[str, Any] = configurable.get("run_config", {})

    gate_fired = not state["subtasks"] and is_simple_task(state["task"])
    if gate_fired:
        state = {
            **state,
            "subtasks": [Subtask("_main", state["task"], [])],
            "plans": {"_main": state["task"]},
        }

    actions = select_action(dict(state))
    if not actions:
        return {"actions": [], "budget_remaining": state["budget_remaining"]}

    temperature = run_config.get("temperature", 0.3)
    adapter_scaling = run_config.get("adapter_scaling", 1.0)

    results: list[tuple[Action, str, str, str | None]] = []
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
        )
        target_name = action.target_subtask or ""
        results.append((action, target_name, result.text, adapter.adapter_id))

    code_actions = [(a, name, text) for a, name, text, _ in results if a.executes_code]
    sandbox_results = await asyncio.gather(
        *[asyncio.to_thread(run_in_sandbox, text) for _, _, text in code_actions]
    )
    feedback_map = {
        name: Feedback(stdout=fb.stdout, stderr=fb.stderr, exit_code=fb.exit_code)
        for (_, name, _), fb in zip(code_actions, sandbox_results, strict=True)
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

    if gate_fired:
        updates.setdefault("subtasks", [Subtask("_main", state["task"], [])])
        if "_main" not in updates.get("plans", {}):
            plans = updates.get("plans", {})
            plans["_main"] = state["task"]
            updates["plans"] = plans

    records = [
        StepRecord(
            step=state["step"],
            action_name=a.name,
            target_subtask=name,
            adapter_id=aid,
            feedback=feedback_map.get(name),
        )
        for a, name, _, aid in results
    ]
    updates["actions"] = actions
    updates["current_adapter"] = (
        results[-1][3] if results else state["current_adapter"]
    )
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
