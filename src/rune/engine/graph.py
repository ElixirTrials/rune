"""Single-node LangGraph engine: step_node + should_continue loop."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rune.engine.parse import parse_output, render_template
from rune.engine.policy import select_action
from rune.engine.state import Action, Feedback, RunState, StepRecord
from rune.sandbox.executor import run_in_sandbox


def state_to_ctx(state: RunState) -> dict[str, Any]:
    """Extract template context variables from RunState."""
    return {
        "task": state["task"],
        "subtasks": state["subtasks"],
        "plans": state["plans"],
        "code": state.get("code_results", {}),
        "integrated_code": state["integrated_code"],
        "feedback": state["feedback"],
        "diagnosis": state["diagnosis"],
        "interfaces": state["interfaces"],
    }


async def step_node(state: RunState, config: RunnableConfig) -> dict[str, Any]:
    """Execute one engine step: select actions, generate, sandbox, parse.

    Args:
        state: Current RunState.
        config: LangGraph RunnableConfig with ``configurable.model`` and
            optional ``configurable.run_config``.

    Returns:
        Partial state update dict with new actions, trajectory, step, and
        budget_remaining.
    """
    configurable: dict[str, Any] = config.get("configurable", {})
    model = configurable["model"]
    run_config: dict[str, Any] = configurable.get("run_config", {})

    actions = select_action(dict(state))
    if not actions:
        return {"actions": [], "budget_remaining": state["budget_remaining"]}

    results: list[tuple[Action, str, str]] = []
    for action in actions:
        ctx = state_to_ctx(state)
        trajectory_text = render_template(action.trajectory_template, **ctx)
        prompt_text = render_template(action.prompt_template, **ctx)

        adapter = model.generate_adapter(trajectory_text)
        model.hotswap_adapter(adapter.state_dict)
        result = await model.generate(
            prompt=prompt_text,
            system_prompt=action.system_prompt,
            output_schema=action.output_schema,
            max_tokens=run_config.get("max_tokens", 2048),
        )
        target_name = action.target_subtask or ""
        results.append((action, target_name, result.text))

    code_actions = [(a, name, text) for a, name, text in results if a.executes_code]
    sandbox_results = await asyncio.gather(
        *[asyncio.to_thread(run_in_sandbox, text) for _, _, text in code_actions]
    )
    feedback_map = {
        name: Feedback(stdout=fb.stdout, stderr=fb.stderr, exit_code=fb.exit_code)
        for (_, name, _), fb in zip(code_actions, sandbox_results, strict=True)
    }

    updates: dict[str, Any] = {}
    for action, target_name, raw in results:
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
            adapter_id=state["current_adapter"],
            feedback=feedback_map.get(name),
        )
        for a, name, _ in results
    ]
    updates["actions"] = actions
    updates["trajectory"] = state["trajectory"] + records
    updates["step"] = state["step"] + 1
    updates["budget_remaining"] = state["budget_remaining"] - 1
    return updates


def should_continue(state: RunState) -> str:
    """Conditional edge: return ``"continue"`` or ``"done"``.

    Returns ``"done"`` when no actions remain or budget is exhausted.
    """
    if not state["actions"] or state["budget_remaining"] <= 0:
        return "done"
    return "continue"


def create_engine() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the single-node LangGraph engine.

    Returns:
        Compiled StateGraph with a ``step`` node and ``should_continue``
        conditional edge looping back to ``step`` or exiting to END.
    """
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
