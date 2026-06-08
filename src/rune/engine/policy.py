"""Deterministic action-selection policy and DAG execution-layer builder."""

from __future__ import annotations

import logging
from dataclasses import replace
from graphlib import CycleError, TopologicalSorter
from typing import Any

from rune.engine.parse import (
    DecomposeResult,
    DiagnoseResult,
    PlanResult,
)
from rune.engine.state import Action, Subtask

logger = logging.getLogger(__name__)

MAX_REPAIRS = 2
MAX_RETRIES = MAX_REPAIRS * 2

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
        "plan",
        "plan",
        "prompt_plan",
        "You are a project planner.",
        PlanResult,
        False,
        None,
    ),
    "code": Action(
        "code",
        "code",
        "prompt_code",
        "You are a code generator.",
        None,  # freeform code (```python fence), de-fenced — never JSON-wrapped
        True,
        None,
    ),
    "repair": Action(
        "repair",
        "code_repair",
        "prompt_code_repair",
        "You are a code generator.",
        None,  # freeform code (```python fence), de-fenced — never JSON-wrapped
        True,
        None,
    ),
    "integrate": Action(
        "integrate",
        "integrate",
        "prompt_integrate",
        "You are a code integrator.",
        None,  # freeform code (```python fence), de-fenced — never JSON-wrapped
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


def build_execution_layers(subtasks: list[Subtask]) -> list[list[str]]:
    known = {s.name for s in subtasks}
    # Restrict edges to known subtasks so a phantom dependency cannot inject an
    # extra node or block readiness.
    graph: dict[str, set[str]] = {
        s.name: {d for d in s.depends_on if d in known} for s in subtasks
    }
    sorter = TopologicalSorter(graph)
    try:
        sorter.prepare()
    except CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else exc
        logger.warning(
            "Cyclic subtask dependencies %s; treating subtasks as independent",
            cycle,
        )
        sorter = TopologicalSorter({name: set() for name in graph})
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
    return replace(ACTIONS[action_name], target_subtask=target)


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
        exhausted: list[str] = []
        for s in ready:
            repairs = state["retries"].get(s.name, 0)
            if repairs >= MAX_RETRIES:
                exhausted.append(s.name)
                continue
            has_code = s.name in state["code_results"]
            has_diagnosis = s.name in state.get("diagnosis", {})

            if not has_code or repairs >= MAX_REPAIRS:
                actions.append(_with_target("code", s.name))
            elif has_diagnosis:
                actions.append(_with_target("repair", s.name))
            else:
                actions.append(_with_target("diagnose", s.name))
        if actions:
            return actions
        if exhausted:
            logger.warning(
                "Subtasks %s exhausted all %d retries",
                exhausted,
                MAX_RETRIES,
            )
            # If integration has already been attempted and is still failing,
            # stop rather than looping integrate<->diagnose until the budget is
            # spent: no repairable work remains.
            int_fb = state.get("integration_feedback")
            if int_fb and int_fb.exit_code != 0:
                logger.warning(
                    "All repairable subtasks exhausted and integration still "
                    "failing; stopping run."
                )
                return []
            entry = str(state.get("entry_point", "") or "")
            if (
                entry
                and len(subtasks) == 1
                and subtasks[0].name == entry
                and str(state.get("public_checks", "") or "").strip()
                and state.get("best_code", {}).get(entry)
            ):
                logger.info(
                    "Single benchmark subtask %s exhausted; shipping best_code",
                    entry,
                )
                return []

    # All subtasks pass — integrate or done
    if len(subtasks) == 1:
        only = subtasks[0].name
        if state["code_passed"].get(only):
            return []

    if state["integrated_code"]:
        return []

    integration_fb = state.get("integration_feedback")
    if integration_fb and integration_fb.exit_code != 0:
        if state.get("diagnosis"):
            return [ACTIONS["integrate"]]
        return [ACTIONS["diagnose"]]
    return [ACTIONS["integrate"]]
