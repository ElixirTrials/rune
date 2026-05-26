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
    """

    name: str
    description: str
    depends_on: list[str]


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


@dataclass(frozen=True)
class StepRecord:
    """Immutable record of one engine step for trajectory logging.

    Attributes:
        step: Zero-based step index within the run.
        action_name: Name of the action executed.
        target_subtask: Subtask targeted by this step, or None.
        adapter_id: ID of the active LoRA adapter, or None.
        feedback: Sandbox result if code was executed, otherwise None.
    """

    step: int
    action_name: str
    target_subtask: str | None
    adapter_id: str | None
    feedback: Feedback | None


class RunState(TypedDict):
    """Full mutable state threaded through the LangGraph engine.

    Attributes:
        task: Top-level task description provided by the user.
        subtasks: Decomposed subtasks.
        interfaces: Mapping of subtask name to extracted code signatures.
        plans: Mapping of subtask name to planning prose.
        code_results: Mapping of subtask name to generated source code.
        code_passed: Mapping of subtask name to sandbox pass/fail.
        retries: Mapping of subtask name to retry count.
        integrated_code: Final merged source after all subtasks pass.
        current_adapter: ID of the active LoRA adapter, or None.
        feedback: Per-subtask sandbox feedback, keyed by subtask name.
        integration_feedback: Sandbox feedback from the integration step, or None.
        diagnosis: Per-subtask fix guidance from diagnose actions, keyed by subtask name.
        actions: Actions selected in the most recent step.
        trajectory: Ordered list of step records for the full run.
        step: Current step index.
        budget_remaining: Steps remaining before forced termination.
    """

    task: str
    subtasks: list[Subtask]
    interfaces: dict[str, str]
    plans: dict[str, str]
    code_results: dict[str, str]
    code_passed: dict[str, bool]
    retries: dict[str, int]
    integrated_code: str
    current_adapter: str | None
    feedback: dict[str, Feedback]
    integration_feedback: Feedback | None
    diagnosis: dict[str, str]
    actions: list[Action]
    trajectory: list[StepRecord]
    step: int
    budget_remaining: int
