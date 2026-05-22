from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchTask:
    task_id: str
    description: str
    test_code: str
    entry_point: str = "solution"


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    code: str
    stderr: str


@dataclass
class BenchResult:
    pass_at_1: float
    total_tasks: int
    passed_tasks: int
    per_task: list[TaskResult] = field(default_factory=list)


def load_tasks(path: Path) -> list[BenchTask]:
    data = json.loads(path.read_text())
    return [BenchTask(**t) for t in data]


def run_benchmark(
    tasks: list[BenchTask],
    engine: Any,
    config: dict[str, Any],
) -> BenchResult:
    results: list[TaskResult] = []
    for task in tasks:
        raise NotImplementedError("Benchmark execution not yet implemented")
    passed = sum(1 for r in results if r.passed)
    return BenchResult(
        pass_at_1=passed / len(results) if results else 0.0,
        total_tasks=len(results),
        passed_tasks=passed,
        per_task=results,
    )
