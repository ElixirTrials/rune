"""Benchmark runner: task loading, execution, and pass@1 scoring."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def run_benchmark(
    tasks: list[BenchTask],
    engine: Any,
    config: dict[str, Any],
) -> BenchResult:
    """Run the full benchmark suite and return aggregate results.

    Args:
        tasks: Tasks to evaluate.
        engine: Compiled LangGraph engine.
        config: Engine run configuration dict.

    Returns:
        BenchResult with pass@1 and per-task details.
    """
    results: list[TaskResult] = []
    for _task in tasks:
        raise NotImplementedError("Benchmark execution not yet implemented")
    passed = sum(1 for r in results if r.passed)
    return BenchResult(
        pass_at_1=passed / len(results) if results else 0.0,
        total_tasks=len(results),
        passed_tasks=passed,
        per_task=results,
    )
