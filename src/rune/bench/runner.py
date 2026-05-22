"""Benchmark runner: task loading, execution, and pass@1 scoring."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rune.sandbox.executor import run_in_sandbox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchTask:
    """A single benchmark problem.

    Attributes:
        task_id: Unique problem identifier.
        description: Natural-language problem statement.
        test_code: Python test code executed against the solution.
        entry_point: Expected function name in the generated solution.
    """

    task_id: str
    description: str
    test_code: str
    entry_point: str = "solution"


@dataclass(frozen=True)
class TaskResult:
    """Result for one benchmark task.

    Attributes:
        task_id: Identifier matching the source BenchTask.
        passed: True if the solution passed all tests.
        code: Generated code that was evaluated.
        stderr: Stderr from the test run.
    """

    task_id: str
    passed: bool
    code: str
    stderr: str


@dataclass
class BenchResult:
    """Aggregate benchmark results.

    Attributes:
        pass_at_1: Fraction of tasks passed on first attempt.
        total_tasks: Total tasks evaluated.
        passed_tasks: Number of tasks that passed.
        per_task: Individual TaskResult for each task.
    """

    pass_at_1: float
    total_tasks: int
    passed_tasks: int
    per_task: list[TaskResult] = field(default_factory=list)


def load_tasks(path: Path) -> list[BenchTask]:
    """Load benchmark tasks from a JSON file.

    Args:
        path: Path to a JSON file containing a list of task dicts.

    Returns:
        List of BenchTask instances.
    """
    data = json.loads(path.read_text())
    return [BenchTask(**t) for t in data]


async def run_benchmark(
    tasks: list[BenchTask],
    engine: Any,
    config: dict[str, Any],
) -> BenchResult:
    """Run the full benchmark suite and return aggregate results.

    Args:
        tasks: Tasks to evaluate.
        engine: Compiled LangGraph engine (CompiledStateGraph).
        config: Configurable dict passed to engine.ainvoke (contains
            ``model`` and ``run_config`` keys).

    Returns:
        BenchResult with pass@1 and per-task details.
    """
    if not tasks:
        return BenchResult(pass_at_1=0.0, total_tasks=0, passed_tasks=0, per_task=[])

    budget = config.get("run_config", {}).get("max_phase_iterations", 5)
    results: list[TaskResult] = []

    for task in tasks:
        initial_state: dict[str, Any] = {
            "task": task.description,
            "subtasks": [],
            "interfaces": {},
            "plans": {},
            "code_results": {},
            "code_passed": {},
            "retries": {},
            "integrated_code": "",
            "current_adapter": None,
            "feedback": None,
            "diagnosis": None,
            "actions": [],
            "trajectory": [],
            "step": 0,
            "budget_remaining": budget,
        }

        try:
            final_state: dict[str, Any] = await engine.ainvoke(
                initial_state, config={"configurable": config}
            )
        except Exception:
            logger.exception("Engine failed for task %s", task.task_id)
            results.append(
                TaskResult(
                    task_id=task.task_id, passed=False, code="", stderr="engine error"
                )
            )
            continue

        generated_code = final_state.get("integrated_code") or ""
        if not generated_code:
            generated_code = "\n".join(final_state.get("code_results", {}).values())

        full_code = generated_code + "\n\n" + task.test_code

        try:
            sandbox_result = await asyncio.to_thread(run_in_sandbox, full_code)
        except Exception:
            logger.exception("Sandbox failed for task %s", task.task_id)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    passed=False,
                    code=generated_code,
                    stderr="sandbox error",
                )
            )
            continue

        passed = sandbox_result.exit_code == 0
        results.append(
            TaskResult(
                task_id=task.task_id,
                passed=passed,
                code=generated_code,
                stderr=sandbox_result.stderr,
            )
        )

    n_passed = sum(1 for r in results if r.passed)
    return BenchResult(
        pass_at_1=n_passed / len(results) if results else 0.0,
        total_tasks=len(results),
        passed_tasks=n_passed,
        per_task=results,
    )
