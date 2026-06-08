"""LangGraph state types for the Rune engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class Subtask:
    """A single unit of work decomposed from the top-level task.

    Attributes:
        name: Unique identifier for the subtask.
        description: Human-readable description of what to implement.
        depends_on: Names of subtasks that must complete before this one.
        acceptance_check: A concrete example I/O or assert for THIS sub-goal —
            the in-loop correctness signal for the subtask's dev cycle.
        builds: The piece of the final entry_point this subtask contributes
            (used to AST-verify integration defines the entry_point).
    """

    name: str
    description: str
    depends_on: list[str]
    acceptance_check: str = ""
    builds: str = ""


@dataclass(frozen=True)
class Action:
    """A parameterized engine action to be executed in a step.

    Attributes:
        name: Action identifier (e.g. "decompose", "code", "integrate").
        trajectory_template: Jinja2 template name for trajectory rendering.
        prompt_template: Jinja2 template name for the model prompt.
        system_prompt: System role string passed to the model.
        output_schema: Pydantic model for structured output, or None for freeform.
        executes_code: Whether the model output is run in the sandbox.
        target_subtask: Name of the subtask this action targets, or None.
    """

    name: str
    trajectory_template: str
    prompt_template: str
    system_prompt: str
    output_schema: type[Any] | None
    executes_code: bool
    target_subtask: str | None


@dataclass(frozen=True)
class Feedback:
    """Result from running generated code in the sandbox.

    Attributes:
        stdout: Standard output captured from the subprocess.
        stderr: Standard error captured from the subprocess.
        exit_code: Process exit code; 0 indicates success.
    """

    stdout: str
    stderr: str
    exit_code: int


_CODE_HISTORY_CAP = 1500


@dataclass(frozen=True)
class StepRecord:
    step: int
    action_name: str
    target_subtask: str | None
    adapter_id: str | None
    feedback: Feedback | None
    generated_code: str | None = None
    trajectory_text: str = ""
    prompt_text: str = ""
    output_text: str = ""


class RunState(TypedDict):
    """Full mutable state threaded through the LangGraph engine.

    Attributes:
        task: Top-level task description provided by the user.
        subtasks: Decomposed subtasks.
        plans: Mapping of subtask name to planning prose.
        code_results: Mapping of subtask name to generated source code.
        code_passed: Mapping of subtask name to sandbox pass/fail.
        code_solved: Subtasks that passed and must not be re-opened by diagnose.
        retries: Mapping of subtask name to retry count.
        integrated_code: Final merged source after all subtasks pass.
        current_adapter: ID of the active LoRA adapter, or None.
        feedback: Per-subtask sandbox feedback, keyed by subtask name.
        integration_feedback: Sandbox feedback from the integration step, or None.
        diagnosis: Per-subtask fix guidance from diagnose actions.
        repair_briefs: Per-subtask deterministic repair brief text.
        plan_attempts: Per-subtask plan-gate rejection count.
        plan_rejections: Per-subtask plan-gate deficiency feedback for replan.
        replan_targets: Subtasks flagged for replan (complexity / escalation).
        max_repairs: Override for repair budget (0 = policy default).
        actions: Actions selected in the most recent step.
        trajectory: Ordered list of step records for the full run.
        step: Current step index.
        budget_remaining: Steps remaining before forced termination.
    """

    task: str
    entry_point: str
    signature: str
    public_checks: str
    overall_goal: str
    subtasks: list[Subtask]
    plans: dict[str, str]
    code_results: dict[str, str]
    code_passed: dict[str, bool]
    code_solved: dict[str, bool]
    # Best candidate seen per subtask, ranked by sandbox quality (pass > assertion
    # mismatch > runtime crash > syntax/empty). The engine ships THIS, never a
    # later worse attempt, so a re-code/repair can't regress a near-miss into a
    # crash and throw away a would-be success (issue #52 RC-C).
    best_code: dict[str, str]
    best_quality: dict[str, int]
    retries: dict[str, int]
    integrated_code: str
    current_adapter: str | None
    feedback: dict[str, Feedback]
    integration_feedback: Feedback | None
    diagnosis: dict[str, str]
    repair_briefs: dict[str, str]
    plan_attempts: dict[str, int]
    plan_rejections: dict[str, str]
    replan_targets: dict[str, bool]
    max_repairs: int
    repair_brief_enabled: bool
    plan_gate_enabled: bool
    replan_on_complexity: bool
    plan_gate_max_attempts: int
    actions: list[Action]
    trajectory: list[StepRecord]
    step: int
    budget_remaining: int


def make_initial_state(
    task: str,
    budget: int,
    entry_point: str = "",
    signature: str = "",
    public_checks: str = "",
    *,
    max_repairs: int = 0,
    repair_brief_enabled: bool = True,
    plan_gate_enabled: bool = True,
    replan_on_complexity: bool = True,
    plan_gate_max_attempts: int = 2,
) -> RunState:
    return {
        "task": task,
        "entry_point": entry_point,
        "signature": signature,
        "public_checks": public_checks,
        "overall_goal": "",
        "subtasks": [],
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "code_solved": {},
        "best_code": {},
        "best_quality": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {},
        "diagnosis": {},
        "repair_briefs": {},
        "plan_attempts": {},
        "plan_rejections": {},
        "replan_targets": {},
        "max_repairs": max_repairs,
        "repair_brief_enabled": repair_brief_enabled,
        "plan_gate_enabled": plan_gate_enabled,
        "replan_on_complexity": replan_on_complexity,
        "plan_gate_max_attempts": plan_gate_max_attempts,
        "integration_feedback": None,
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": budget,
    }
