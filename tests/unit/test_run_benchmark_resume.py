from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rune.bench.runner import GRADING_GATE_VERSION, BenchTask, run_benchmark


class _ExplodingEngine:
    """Fails the test if the engine is invoked for a task that should be resumed."""

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("engine must not run a resumed task")


class _RecordingEngine:
    """Records that a stale-stamped task was re-run rather than resumed."""

    def __init__(self) -> None:
        self.ran = False

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.ran = True
        return {**state, "best_code": {}, "integrated_code": ""}


def _write_meta(sess: Path, task_id: str, meta: dict[str, Any]) -> None:
    (sess / task_id).mkdir(parents=True)
    (sess / task_id / "metadata.json").write_text(json.dumps(meta))


def test_resume_skips_completed_task(tmp_path: Path) -> None:
    task = BenchTask(task_id="HumanEval/0", description="d", test_code="assert True", entry_point="f")
    sess = tmp_path / "sessions"
    _write_meta(
        sess,
        task.task_id,
        {"pass_at_1": True, "grading_gate_version": GRADING_GATE_VERSION},
    )

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


def test_resume_reruns_unstamped_metadata(tmp_path: Path) -> None:
    # Pre-fix labels carry no grading_gate_version; they must NOT be re-served.
    task = BenchTask(task_id="HumanEval/0", description="d", test_code="assert True", entry_point="f")
    sess = tmp_path / "sessions"
    _write_meta(sess, task.task_id, {"pass_at_1": True})
    engine = _RecordingEngine()

    asyncio.run(
        run_benchmark(
            [task],
            engine,
            {"run_config": {"max_phase_iterations": 3}},
            sessions_dir=sess,
            resume=True,
        )
    )
    assert engine.ran is True


def test_resume_reruns_stale_stamp(tmp_path: Path) -> None:
    task = BenchTask(task_id="HumanEval/0", description="d", test_code="assert True", entry_point="f")
    sess = tmp_path / "sessions"
    _write_meta(
        sess,
        task.task_id,
        {"pass_at_1": True, "grading_gate_version": GRADING_GATE_VERSION - 1},
    )
    engine = _RecordingEngine()

    asyncio.run(
        run_benchmark(
            [task],
            engine,
            {"run_config": {"max_phase_iterations": 3}},
            sessions_dir=sess,
            resume=True,
        )
    )
    assert engine.ran is True
