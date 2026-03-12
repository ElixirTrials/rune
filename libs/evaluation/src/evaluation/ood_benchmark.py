"""Out-of-distribution benchmark for adapter generalization testing.

Provides functions for evaluating adapter performance on tasks outside
the training distribution, measuring generalization capability.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_OOD_DATA_DIR = Path(__file__).parent / "data"


def run_ood_benchmark(
    adapter_id: str | None,
    completions: dict[str, str],
    benchmark_name: str = "ood_python",
) -> dict[str, Any]:
    """Run an out-of-distribution benchmark on provided completions.

    Evaluates completions against OOD tasks from the bundled task set.
    Each task's prompt + completion is executed in a subprocess with its
    test harness.

    Args:
        adapter_id: UUID of the adapter being tested (informational).
        completions: Dict mapping task_id to completion string.
        benchmark_name: Name of the OOD benchmark set.

    Returns:
        Dictionary with ood_pass_rate and per-task results.
    """
    ood_path = _OOD_DATA_DIR / "ood_tasks.json"
    with ood_path.open() as f:
        all_tasks: list[dict[str, str]] = json.load(f)

    task_map = {t["task_id"]: t for t in all_tasks}
    task_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for task_id, completion in completions.items():
            task = task_map.get(task_id)
            if task is None:
                continue

            script = task["prompt"] + completion + "\n" + task["test"] + "\n"
            script_path = Path(tmpdir) / f"{task_id.replace('/', '_')}.py"
            script_path.write_text(script)

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir,
                )
                passed = proc.returncode == 0
            except subprocess.TimeoutExpired:
                passed = False

            task_results.append({"task_id": task_id, "passed": passed})

    total = len(task_results)
    pass_count = sum(1 for r in task_results if r["passed"])
    ood_pass_rate = pass_count / total if total > 0 else 0.0

    return {
        "adapter_id": adapter_id,
        "benchmark_name": benchmark_name,
        "ood_pass_rate": ood_pass_rate,
        "pass_count": pass_count,
        "total": total,
        "task_results": task_results,
    }


def compute_generalization_delta(
    in_dist_rate: float,
    ood_rate: float,
) -> float:
    """Compute the generalization delta between in-distribution and OOD performance.

    A positive delta means the adapter does better on OOD tasks relative to
    in-distribution performance. A negative delta indicates overfitting.

    Args:
        in_dist_rate: Pass rate on in-distribution tasks (0.0 to 1.0).
        ood_rate: Pass rate on OOD tasks (0.0 to 1.0).

    Returns:
        Generalization delta as ood_rate - in_dist_rate.
    """
    return ood_rate - in_dist_rate
