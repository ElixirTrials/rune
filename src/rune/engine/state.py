"""LangGraph state types for the Rune engine."""

from __future__ import annotations

from collections.abc import Mapping
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
    advisory_requirement_kinds: tuple[str, ...]
    constraint_scale_pass_quality: bool
    complexity_probe_min_n: int
    complexity_probe_max_n: int
    complexity_probe_n_repeats: int
    complexity_probe_per_run_timeout_s: float
    complexity_empirical_timeout_s: float
    complexity_judge_enabled: bool
    complexity_judge_temperature: float
    complexity_judge_max_tokens: int
    merge_spec_public_checks: bool
    ship_best_on_exhaustion: bool
    ship_best_min_quality: int
    complexity_repair_preserve_logic: bool
    actions: list[Action]
    trajectory: list[StepRecord]
    step: int
    budget_remaining: int


def engine_kwargs_from_run_config(
    run_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Engine flags from ``run_config`` / :class:`~rune.config.PipelineConfig`.

    Defaults come from :func:`rune.config.load_rune_config` so config.yaml is the
    single source of truth; explicit ``run_config`` keys override.
    """
    from rune.config import PipelineConfig  # noqa: PLC0415

    defaults = PipelineConfig()
    rc = run_config or {}
    kinds = rc.get("advisory_requirement_kinds", defaults.advisory_requirement_kinds)
    return {
        "max_repairs": int(rc.get("max_repairs", defaults.max_repairs) or 0),
        "repair_brief_enabled": bool(
            rc.get("repair_brief_enabled", defaults.repair_brief_enabled)
        ),
        "plan_gate_enabled": bool(
            rc.get("plan_gate_enabled", defaults.plan_gate_enabled)
        ),
        "replan_on_complexity": bool(
            rc.get("replan_on_complexity", defaults.replan_on_complexity)
        ),
        "plan_gate_max_attempts": int(
            rc.get("plan_gate_max_attempts", defaults.plan_gate_max_attempts)
        ),
        "advisory_requirement_kinds": tuple(kinds),
        "constraint_scale_pass_quality": bool(
            rc.get(
                "constraint_scale_pass_quality",
                defaults.constraint_scale_pass_quality,
            )
        ),
        "complexity_probe_min_n": int(
            rc.get("complexity_probe_min_n", defaults.complexity_probe_min_n)
        ),
        "complexity_probe_max_n": int(
            rc.get("complexity_probe_max_n", defaults.complexity_probe_max_n)
        ),
        "complexity_probe_n_repeats": int(
            rc.get("complexity_probe_n_repeats", defaults.complexity_probe_n_repeats)
        ),
        "complexity_probe_per_run_timeout_s": float(
            rc.get(
                "complexity_probe_per_run_timeout_s",
                defaults.complexity_probe_per_run_timeout_s,
            )
        ),
        "complexity_empirical_timeout_s": float(
            rc.get(
                "complexity_empirical_timeout_s",
                defaults.complexity_empirical_timeout_s,
            )
        ),
        "complexity_judge_enabled": bool(
            rc.get("complexity_judge_enabled", defaults.complexity_judge_enabled)
        ),
        "complexity_judge_temperature": float(
            rc.get(
                "complexity_judge_temperature",
                defaults.complexity_judge_temperature,
            )
        ),
        "complexity_judge_max_tokens": int(
            rc.get(
                "complexity_judge_max_tokens",
                defaults.complexity_judge_max_tokens,
            )
        ),
        "merge_spec_public_checks": bool(
            rc.get("merge_spec_public_checks", defaults.merge_spec_public_checks)
        ),
        "ship_best_on_exhaustion": bool(
            rc.get("ship_best_on_exhaustion", defaults.ship_best_on_exhaustion)
        ),
        "ship_best_min_quality": int(
            rc.get("ship_best_min_quality", defaults.ship_best_min_quality)
        ),
        "complexity_repair_preserve_logic": bool(
            rc.get(
                "complexity_repair_preserve_logic",
                defaults.complexity_repair_preserve_logic,
            )
        ),
    }


def advisory_kinds_from_state(state: Mapping[str, Any]) -> frozenset[str]:
    """Advisory requirement kinds configured for this run."""
    from rune.config import PipelineConfig  # noqa: PLC0415

    default = PipelineConfig().advisory_requirement_kinds
    return frozenset(state.get("advisory_requirement_kinds", default))


def make_initial_state(
    task: str,
    budget: int,
    entry_point: str = "",
    signature: str = "",
    public_checks: str = "",
    *,
    run_config: Mapping[str, Any] | None = None,
    max_repairs: int | None = None,
    repair_brief_enabled: bool | None = None,
    plan_gate_enabled: bool | None = None,
    replan_on_complexity: bool | None = None,
    plan_gate_max_attempts: int | None = None,
) -> RunState:
    engine = engine_kwargs_from_run_config(run_config)
    if max_repairs is not None:
        engine["max_repairs"] = max_repairs
    if repair_brief_enabled is not None:
        engine["repair_brief_enabled"] = repair_brief_enabled
    if plan_gate_enabled is not None:
        engine["plan_gate_enabled"] = plan_gate_enabled
    if replan_on_complexity is not None:
        engine["replan_on_complexity"] = replan_on_complexity
    if plan_gate_max_attempts is not None:
        engine["plan_gate_max_attempts"] = plan_gate_max_attempts
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
        "integration_feedback": None,
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": budget,
        "max_repairs": int(engine["max_repairs"]),
        "repair_brief_enabled": bool(engine["repair_brief_enabled"]),
        "plan_gate_enabled": bool(engine["plan_gate_enabled"]),
        "replan_on_complexity": bool(engine["replan_on_complexity"]),
        "plan_gate_max_attempts": int(engine["plan_gate_max_attempts"]),
        "advisory_requirement_kinds": tuple(engine["advisory_requirement_kinds"]),
        "constraint_scale_pass_quality": bool(engine["constraint_scale_pass_quality"]),
        "complexity_probe_min_n": int(engine["complexity_probe_min_n"]),
        "complexity_probe_max_n": int(engine["complexity_probe_max_n"]),
        "complexity_probe_n_repeats": int(engine["complexity_probe_n_repeats"]),
        "complexity_probe_per_run_timeout_s": float(
            engine["complexity_probe_per_run_timeout_s"]
        ),
        "complexity_empirical_timeout_s": float(
            engine["complexity_empirical_timeout_s"]
        ),
        "complexity_judge_enabled": bool(engine["complexity_judge_enabled"]),
        "complexity_judge_temperature": float(engine["complexity_judge_temperature"]),
        "complexity_judge_max_tokens": int(engine["complexity_judge_max_tokens"]),
        "merge_spec_public_checks": bool(engine["merge_spec_public_checks"]),
        "ship_best_on_exhaustion": bool(engine["ship_best_on_exhaustion"]),
        "ship_best_min_quality": int(engine["ship_best_min_quality"]),
        "complexity_repair_preserve_logic": bool(
            engine["complexity_repair_preserve_logic"]
        ),
    }
