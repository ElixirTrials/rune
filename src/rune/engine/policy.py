from __future__ import annotations

from graphlib import TopologicalSorter

from rune.engine.parse import DecomposeResult, DiagnoseResult
from rune.engine.state import Action, Subtask

MAX_RETRIES = 3

ACTIONS: dict[str, Action] = {
    "decompose": Action("decompose", "decompose", "prompt_decompose", "You are a project decomposer.", DecomposeResult, False, None),
    "plan": Action("plan", "plan", "prompt_plan", "You are a project planner.", None, False, None),
    "code": Action("code", "code", "prompt_code", "You are a code generator.", None, True, None),
    "code_retry": Action("code_retry", "code_retry", "prompt_code_retry", "You are a code generator.", None, True, None),
    "integrate": Action("integrate", "integrate", "prompt_integrate", "You are a code integrator.", None, True, None),
    "diagnose": Action("diagnose", "diagnose", "prompt_diagnose", "You are a code diagnostician.", DiagnoseResult, False, None),
}


def build_execution_layers(subtasks: list[Subtask]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for s in subtasks:
        graph[s.name] = set(s.depends_on)
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    layers: list[list[str]] = []
    while sorter.is_active():
        ready = sorted(sorter.get_ready())
        layers.append(ready)
        for node in ready:
            sorter.done(node)
    return layers


def _with_target(action_name: str, target: str) -> Action:
    base = ACTIONS[action_name]
    return Action(
        name=base.name,
        trajectory_template=base.trajectory_template,
        prompt_template=base.prompt_template,
        system_prompt=base.system_prompt,
        output_schema=base.output_schema,
        executes_code=base.executes_code,
        target_subtask=target,
    )


def select_action(state: dict) -> list[Action]:
    subtasks: list[Subtask] = state["subtasks"]
    if not subtasks:
        return [ACTIONS["decompose"]]

    # Plan unplanned subtasks
    unplanned = [s for s in subtasks if s.name not in state["plans"]]
    if unplanned:
        layers = build_execution_layers(unplanned)
        return [_with_target("plan", name) for name in layers[0]]

    # Code uncoded or failing subtasks
    failing = [
        s for s in subtasks
        if not state["code_passed"].get(s.name)
    ]
    if failing:
        layers = build_execution_layers(failing)
        ready_names = set(layers[0])
        # Only subtasks whose deps all pass
        ready = [
            s for s in failing
            if s.name in ready_names
            and all(state["code_passed"].get(d, False) for d in s.depends_on)
        ]
        actions: list[Action] = []
        for s in ready:
            if state["retries"].get(s.name, 0) >= MAX_RETRIES:
                return []  # stuck
            action_name = "code_retry" if s.name in state["code_results"] else "code"
            actions.append(_with_target(action_name, s.name))
        return actions if actions else []

    # All passing — integrate or done
    if state["integrated_code"]:
        return []

    if state.get("diagnosis"):
        return [ACTIONS["integrate"]]
    if state.get("feedback") and state["feedback"].exit_code != 0:
        return [ACTIONS["diagnose"]]
    return [ACTIONS["integrate"]]
