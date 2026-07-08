"""Deterministic action-selection policy and DAG execution-layer builder."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from graphlib import CycleError, TopologicalSorter
from typing import Any

from rune.engine.parse import (
    DecomposeResult,
    DiagnoseResult,
    PlanResult,
    approach_signature,
)
from rune.engine.requirements import is_constraint_scale_only_failure
from rune.engine.state import Action, Subtask

logger = logging.getLogger(__name__)

MAX_REPAIRS = 4
MAX_RETRIES = MAX_REPAIRS * 2

# Volatile substrings masked before comparing two attempts' probe stderr: memory
# addresses, temp-file paths, traceback line numbers, and timing measurements all
# vary run-to-run without reflecting a different bug. What survives — the
# exception type, the actual-vs-expected assertion text, the constraint-scale
# verdict — is the stable failure identity the model must change to progress.
_HEX_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_PY_PATH = re.compile(r"(?:/[^\s\"':]+)+\.py")
_LINE_NO = re.compile(r"line \d+")
_SECONDS = re.compile(r"\d+\.\d+\s*s\b")


def _normalize_stderr(stderr: str) -> str:
    """Canonicalize a probe stderr for cross-attempt equality (see module note)."""
    text = (stderr or "").strip()
    text = _HEX_ADDR.sub("0xADDR", text)
    text = _PY_PATH.sub("<path>", text)
    text = _LINE_NO.sub("line N", text)
    text = _SECONDS.sub("Ns", text)
    return re.sub(r"\s+", " ", text)


def _failed_code_attempts(
    state: dict[str, Any], name: str
) -> list[tuple[str, str, str]]:
    """Failing code/repair attempts for ``name`` in run order (oldest→newest).

    Each tuple is (normalized_stderr, approach_signature, raw_stderr), read from
    the run trajectory — signals already legitimately in engine state, never
    anything derived from held-out tests.
    """
    out: list[tuple[str, str, str]] = []
    for rec in state.get("trajectory", []):
        if getattr(rec, "target_subtask", None) != name:
            continue
        if getattr(rec, "action_name", None) not in ("code", "repair"):
            continue
        fb = getattr(rec, "feedback", None)
        if fb is None or fb.exit_code == 0:
            continue
        code = getattr(rec, "generated_code", None) or ""
        out.append((_normalize_stderr(fb.stderr), approach_signature(code), fb.stderr))
    return out


def _dedup_exhausted(state: dict[str, Any], name: str) -> bool:
    """Same-failure dedup (issue #52 §4 lever 3a). OFF unless repair_dedup_after set.

    Fires when the last ``repair_dedup_after`` failing attempts for the subtask
    all share the same (normalized stderr, approach signature) pair — the model
    is re-submitting an equivalent candidate that cannot make progress. Floored
    at 2: a window of 1 is trivially all-equal and would kill every first
    repair (e.g. 3799's genuine diagnose→repair progression).
    """
    n = state.get("repair_dedup_after")
    if not isinstance(n, int) or n < 1:
        return False
    n = max(n, 2)
    attempts = _failed_code_attempts(state, name)
    if len(attempts) < n:
        return False
    window = attempts[-n:]
    first = (window[0][0], window[0][1])
    return all((a[0], a[1]) == first for a in window)


def _complexity_cap_exhausted(state: dict[str, Any], name: str) -> bool:
    """Complexity-repair cap (issue #52 §4 lever 3b). OFF unless the cap is set.

    Fires when the last ``complexity_repair_cap`` failing attempts for the
    subtask are ALL constraint-scale-only rejections. Such candidates already
    ship at quality 3, and the model reliably re-submits the same brute force.
    """
    k = state.get("complexity_repair_cap")
    if not isinstance(k, int) or k < 1:
        return False
    attempts = _failed_code_attempts(state, name)
    if len(attempts) < k:
        return False
    window = attempts[-k:]
    return all(is_constraint_scale_only_failure(raw) for _, _, raw in window)


def _budget_guard_exhausted(state: dict[str, Any], name: str) -> bool:
    """True when a flag-gated budget guard says to stop retrying ``name``."""
    return _dedup_exhausted(state, name) or _complexity_cap_exhausted(state, name)


def _max_repairs(state: dict[str, Any]) -> int:
    override = int(state.get("max_repairs") or 0)
    return override if override > 0 else MAX_REPAIRS


def _max_retries(state: dict[str, Any]) -> int:
    return _max_repairs(state) * 2


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

    # Handle uncoded or failing subtasks (skip permanently solved)
    code_solved = state.get("code_solved", {})
    failing = [
        s
        for s in subtasks
        if not state["code_passed"].get(s.name) and not code_solved.get(s.name)
    ]
    if failing:
        layers = build_execution_layers(failing)
        ready_names = set(layers[0])
        ready = [
            s
            for s in failing
            if s.name in ready_names
            and all(state["code_passed"].get(d, False) for d in s.depends_on)
        ]
        replan = [
            s
            for s in ready
            if state.get("replan_targets", {}).get(s.name)
            and s.name not in state.get("plans", {})
        ]
        if replan:
            return [_with_target("plan", replan[0].name)]
        max_repairs = _max_repairs(state)
        max_retries = _max_retries(state)
        actions: list[Action] = []
        exhausted: list[str] = []
        for s in ready:
            repairs = state["retries"].get(s.name, 0)
            if repairs >= max_retries:
                exhausted.append(s.name)
                continue
            # Flag-gated budget guards: stop retrying a subtask the model can only
            # re-submit unchanged (same-failure dedup / complexity-only cap) and
            # let the existing exhaustion / ship-best path take over. Both default
            # OFF, so default action selection is bit-identical (issue #52 §4).
            if _budget_guard_exhausted(state, s.name):
                logger.info(
                    "Subtask %s halted by budget guard; treating as exhausted",
                    s.name,
                )
                exhausted.append(s.name)
                continue
            has_code = s.name in state["code_results"]
            has_diagnosis = s.name in state.get("diagnosis", {})
            # A non-empty deterministic repair_brief already carries the
            # structured failure signal, so skip the redundant diagnose step.
            has_brief = bool(state.get("repair_briefs", {}).get(s.name, "").strip())

            if not has_code or repairs >= max_repairs:
                actions.append(_with_target("code", s.name))
            elif has_diagnosis or has_brief:
                actions.append(_with_target("repair", s.name))
            else:
                actions.append(_with_target("diagnose", s.name))
        if actions:
            return actions
        if exhausted:
            logger.warning(
                "Subtasks %s exhausted all %d retries",
                exhausted,
                max_retries,
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
