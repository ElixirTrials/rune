"""Deterministic action-selection policy and DAG execution-layer builder."""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Any

from rune.engine.parse import DecomposeResult, DiagnoseResult
from rune.engine.state import Action, Subtask

MAX_REPAIRS = 2
MAX_RETRIES = MAX_REPAIRS * 2
_SIMPLE_WORD_THRESHOLD = 200
_SIMPLE_SIGNALS = [
    "write a function",
    "implement a function",
    "create a function",
    "write a method",
    "implement a method",
    "write a class",
    "implement a class",
    "create a class",
]

ACTIONS: dict[str, Action] = {
    "decompose": Action(
        "decompose",
        "decompose",
        "prompt_decompose_concise",
        "You are a project decomposer.",
        DecomposeResult,
        False,
        None,
    ),
    "plan": Action(
        "plan", "plan", "prompt_plan", "You are a project planner.", None, False, None
    ),
    "code": Action(
        "code", "code", "prompt_code", "You are a code generator.", None, True, None
    ),
    "repair": Action(
        "repair",
        "code_repair",
        "prompt_code_repair",
        "You are a code generator.",
        None,
        True,
        None,
    ),
    "integrate": Action(
        "integrate",
        "integrate",
        "prompt_integrate",
        "You are a code integrator.",
        None,
        True,
        None,
    ),
    "diagnose": Action(
        "diagnose",
        "diagnose",
        "prompt_diagnose",
        "You are a code diagnostician.",
        DiagnoseResult,
        False,
        None,
    ),
}


def is_simple_task(task: str) -> bool:
    if len(task.split()) > _SIMPLE_WORD_THRESHOLD:
        return False
    task_lower = task.lower()
    return any(signal in task_lower for signal in _SIMPLE_SIGNALS)


def build_execution_layers(subtasks: list[Subtask]) -> list[list[str]]:
    known = {s.name for s in subtasks}
    graph: dict[str, set[str]] = {}
    for s in subtasks:
        graph[s.name] = set(s.depends_on)
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    layers: list[list[str]] = []
    while sorter.is_active():
        batch = sorter.get_ready()
        real = sorted(n for n in batch if n in known)
        if real:
            layers.append(real)
        for node in batch:
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


def select_action(state: dict[str, Any]) -> list[Action]:
    subtasks: list[Subtask] = state["subtasks"]
    if not subtasks:
        return [ACTIONS["decompose"]]

    # Plan unplanned subtasks
    unplanned = [s for s in subtasks if s.name not in state["plans"]]
    if unplanned:
        layers = build_execution_layers(unplanned)
        return [_with_target("plan", name) for name in layers[0]]

    # Handle uncoded or failing subtasks
    failing = [s for s in subtasks if not state["code_passed"].get(s.name)]
    if failing:
        layers = build_execution_layers(failing)
        ready_names = set(layers[0])
        ready = [
            s
            for s in failing
            if s.name in ready_names
            and all(state["code_passed"].get(d, False) for d in s.depends_on)
        ]
        actions: list[Action] = []
        for s in ready:
            repairs = state["retries"].get(s.name, 0)
            if repairs >= MAX_RETRIES:
                continue
            has_code = s.name in state["code_results"]
            has_diagnosis = s.name in state.get("diagnosis", {})

            if not has_code or repairs >= MAX_REPAIRS:
                actions.append(_with_target("code", s.name))
            elif has_diagnosis:
                actions.append(_with_target("repair", s.name))
            else:
                actions.append(_with_target("diagnose", s.name))
        return actions if actions else []

    # All subtasks pass — integrate or done
    if state["integrated_code"]:
        return []

    integration_fb = state.get("integration_feedback")
    if integration_fb and integration_fb.exit_code != 0:
        if state.get("diagnosis"):
            return [ACTIONS["integrate"]]
        return [ACTIONS["diagnose"]]
    return [ACTIONS["integrate"]]
