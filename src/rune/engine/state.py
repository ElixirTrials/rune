from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class Subtask:
    name: str
    description: str
    depends_on: list[str]


@dataclass(frozen=True)
class Action:
    name: str
    trajectory_template: str
    prompt_template: str
    system_prompt: str
    output_schema: type[Any] | None
    executes_code: bool
    target_subtask: str | None


@dataclass(frozen=True)
class Feedback:
    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class StepRecord:
    step: int
    action_name: str
    target_subtask: str | None
    adapter_id: str | None
    feedback: Feedback | None


class RunState(TypedDict):
    task: str
    subtasks: list[Subtask]
    interfaces: dict[str, str]
    plans: dict[str, str]
    code_results: dict[str, str]
    code_passed: dict[str, bool]
    retries: dict[str, int]
    integrated_code: str
    current_adapter: str | None
    feedback: Feedback | None
    diagnosis: str | None
    actions: list[Action]
    trajectory: list[StepRecord]
    step: int
    budget_remaining: int
