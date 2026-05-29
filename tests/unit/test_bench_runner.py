"""Unit tests for the benchmark runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune.bench.runner import (
    BenchResult,
    BenchTask,
    TaskResult,
    load_tasks,
    run_benchmark,
)


def _bench_config(budget: int = 10) -> dict:  # type: ignore[type-arg]
    return {"run_config": {"max_phase_iterations": budget}}


def _make_task(
    task_id: str = "t1",
    description: str = "write add",
    test_code: str = "assert add(1,2)==3",
) -> BenchTask:
    return BenchTask(task_id=task_id, description=description, test_code=test_code)


def _make_engine(final_state: dict) -> AsyncMock:  # type: ignore[type-arg]
    engine = MagicMock()
    engine.ainvoke = AsyncMock(return_value=final_state)
    return engine


def _sandbox_pass() -> MagicMock:
    r = MagicMock()
    r.exit_code = 0
    r.stderr = ""
    return r


def _sandbox_fail() -> MagicMock:
    r = MagicMock()
    r.exit_code = 1
    r.stderr = "AssertionError"
    return r


class TestRunBenchmarkAllPass:
    def test_all_tasks_pass(self) -> None:
        tasks = [_make_task("t1"), _make_task("t2")]
        final_state = {
            "integrated_code": "def add(a,b): return a+b",
            "code_results": {},
        }
        engine = _make_engine(final_state)

        with patch("rune.bench.runner.run_in_sandbox", return_value=_sandbox_pass()):
            result = asyncio.run(run_benchmark(tasks, engine, _bench_config()))

        assert result.pass_at_1 == 1.0
        assert result.total_tasks == 2
        assert result.passed_tasks == 2
        assert all(r.passed for r in result.per_task)


class TestRunBenchmarkPartialPass:
    def test_some_tasks_fail(self) -> None:
        tasks = [_make_task("t1"), _make_task("t2")]
        final_state = {
            "integrated_code": "def add(a,b): return a+b",
            "code_results": {},
        }
        engine = _make_engine(final_state)

        sandbox_results = [_sandbox_pass(), _sandbox_fail()]
        with patch("rune.bench.runner.run_in_sandbox", side_effect=sandbox_results):
            result = asyncio.run(run_benchmark(tasks, engine, _bench_config()))

        assert result.pass_at_1 == pytest.approx(0.5)
        assert result.total_tasks == 2
        assert result.passed_tasks == 1


class TestRunBenchmarkEmptyTasks:
    def test_empty_tasks_returns_zero(self) -> None:
        engine = _make_engine({})
        result = asyncio.run(run_benchmark([], engine, {}))

        assert result.pass_at_1 == 0.0
        assert result.total_tasks == 0
        assert result.passed_tasks == 0
        assert result.per_task == []


class TestRunBenchmarkCodeExtraction:
    def test_uses_integrated_code_when_present(self) -> None:
        task = _make_task("t1", test_code="assert solution() == 42")
        final_state = {
            "integrated_code": "def solution(): return 42",
            "code_results": {"sub": "ignored"},
        }
        engine = _make_engine(final_state)

        captured: list[str] = []

        def capture_sandbox(code: str) -> MagicMock:
            captured.append(code)
            return _sandbox_pass()

        with patch("rune.bench.runner.run_in_sandbox", side_effect=capture_sandbox):
            asyncio.run(run_benchmark([task], engine, _bench_config()))

        assert "def solution(): return 42" in captured[0]
        assert "assert solution() == 42" in captured[0]

    def test_falls_back_to_code_results_when_integrated_empty(self) -> None:
        task = _make_task("t1", test_code="assert add(1,2)==3")
        final_state = {
            "integrated_code": "",
            "code_results": {"sub1": "def add(a,b): return a+b"},
        }
        engine = _make_engine(final_state)

        captured: list[str] = []

        def capture_sandbox(code: str) -> MagicMock:
            captured.append(code)
            return _sandbox_pass()

        with patch("rune.bench.runner.run_in_sandbox", side_effect=capture_sandbox):
            asyncio.run(run_benchmark([task], engine, _bench_config()))

        assert "def add(a,b): return a+b" in captured[0]
        assert "assert add(1,2)==3" in captured[0]

    def test_no_code_at_all_still_runs_test(self) -> None:
        task = _make_task("t1", test_code="assert True")
        final_state = {"integrated_code": "", "code_results": {}}
        engine = _make_engine(final_state)

        with patch(
            "rune.bench.runner.run_in_sandbox", return_value=_sandbox_fail()
        ) as mock_sb:
            result = asyncio.run(run_benchmark([task], engine, _bench_config()))

        assert result.passed_tasks == 0
        mock_sb.assert_called_once()


class TestRunBenchmarkSandboxTimeout:
    def test_timeout_counts_as_fail(self) -> None:
        task = _make_task("t1")
        final_state = {
            "integrated_code": "def add(a,b): return a+b",
            "code_results": {},
        }
        engine = _make_engine(final_state)

        timeout_result = MagicMock()
        timeout_result.exit_code = -1
        timeout_result.stderr = "Timeout"

        with patch("rune.bench.runner.run_in_sandbox", return_value=timeout_result):
            result = asyncio.run(run_benchmark([task], engine, _bench_config()))

        assert result.passed_tasks == 0
        assert not result.per_task[0].passed


class TestRunBenchmarkConfigPropagation:
    def test_budget_comes_from_config(self) -> None:
        task = _make_task("t1")
        final_state = {"integrated_code": "x=1", "code_results": {}}
        engine = _make_engine(final_state)

        config = {"run_config": {"max_phase_iterations": 7}, "model": MagicMock()}
        captured_states: list[dict] = []  # type: ignore[type-arg]

        async def capture_ainvoke(state: dict, config: dict) -> dict:  # type: ignore[type-arg]
            captured_states.append(state)
            return final_state

        engine.ainvoke = capture_ainvoke

        with patch("rune.bench.runner.run_in_sandbox", return_value=_sandbox_pass()):
            asyncio.run(run_benchmark([task], engine, config))

        assert captured_states[0]["budget_remaining"] == 7


class TestLoadTasks:
    def test_load_tasks_from_json(self, tmp_path: Path) -> None:
        data = [
            {
                "task_id": "a",
                "description": "do a",
                "test_code": "assert True",
                "entry_point": "solution",
            },
        ]
        p = tmp_path / "tasks.json"
        p.write_text(json.dumps(data))
        tasks = load_tasks(p)
        assert len(tasks) == 1
        assert tasks[0].task_id == "a"


class TestTaskResultDataclass:
    def test_task_result_fields(self) -> None:
        r = TaskResult(task_id="x", passed=True, code="x=1", stderr="")
        assert r.task_id == "x"
        assert r.passed is True


class TestBenchResultDataclass:
    def test_bench_result_fields(self) -> None:
        r = BenchResult(pass_at_1=0.75, total_tasks=4, passed_tasks=3)
        assert r.pass_at_1 == 0.75
        assert r.per_task == []
