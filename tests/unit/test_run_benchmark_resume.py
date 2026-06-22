from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rune.bench.runner import BenchTask, run_benchmark


class _ExplodingEngine:
    """Fails the test if the engine is invoked for a task that should be resumed."""

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("engine must not run a resumed task")


def test_resume_skips_completed_task(tmp_path: Path) -> None:
    task = BenchTask(task_id="HumanEval/0", description="d", test_code="assert True", entry_point="f")
    sess = tmp_path / "sessions"
    (sess / task.task_id).mkdir(parents=True)
    (sess / task.task_id / "metadata.json").write_text(json.dumps({"pass_at_1": True}))

    result = asyncio.run(
        run_benchmark(
            [task],
            _ExplodingEngine(),
            {"run_config": {"max_phase_iterations": 3}},
            sessions_dir=sess,
            resume=True,
        )
    )
    assert result.passed_tasks == 1
    assert result.per_task[0].passed is True
