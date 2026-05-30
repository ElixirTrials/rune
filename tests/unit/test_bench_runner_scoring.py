"""Bench scoring: the model's own self-tests must not contaminate held-out scoring."""

from __future__ import annotations

import asyncio
from typing import Any

import rune.bench.runner as runner_mod
from rune.bench.runner import BenchTask, run_benchmark


class _FakeEngine:
    """Returns a fixed final_state from ainvoke, ignoring the input state."""

    def __init__(self, integrated_code: str) -> None:
        self._code = integrated_code

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return {"integrated_code": self._code, "code_results": {}}


def _run(tasks: list[BenchTask], engine: Any) -> Any:
    config = {"run_config": {"max_phase_iterations": 3}}
    return asyncio.run(run_benchmark(tasks, engine, config))


def test_correct_impl_with_wrong_self_test_passes_after_strip() -> None:
    # Correct impl, but the model appended a WRONG __main__ self-test.
    code = (
        "def is_num_decagonal(n):\n"
        "    return 4 * n**2 - 3 * n\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    assert is_num_decagonal(10) == 380  # wrong: actual is 370\n"
    )
    # Held-out test is correct: is_num_decagonal(7) == 175.
    task = BenchTask(
        task_id="decagonal",
        description="nth decagonal number",
        test_code="assert is_num_decagonal(7) == 175\n",
        entry_point="is_num_decagonal",
    )
    result = _run([task], _FakeEngine(code))
    assert result.passed_tasks == 1
    assert result.per_task[0].passed is True


def test_genuinely_wrong_impl_still_fails() -> None:
    # No self-tests; impl is wrong. Must still fail (stripping changes nothing).
    code = "def is_num_decagonal(n):\n    return n\n"
    task = BenchTask(
        task_id="decagonal-bad",
        description="nth decagonal number",
        test_code="assert is_num_decagonal(7) == 175\n",
        entry_point="is_num_decagonal",
    )
    result = _run([task], _FakeEngine(code))
    assert result.passed_tasks == 0
    assert result.per_task[0].passed is False


def test_runner_seeds_rng_per_task(monkeypatch: Any) -> None:
    calls: list[int] = []
    monkeypatch.setattr(runner_mod, "_seed_rng", calls.append)

    tasks = [
        BenchTask(task_id=f"t{i}", description="d", test_code="assert True\n")
        for i in range(3)
    ]
    config = {"run_config": {"max_phase_iterations": 1, "seed": 100}}
    asyncio.run(run_benchmark(tasks, _FakeEngine("x = 1\n"), config))

    assert calls == [100, 101, 102]


def test_runner_does_not_seed_when_seed_absent(monkeypatch: Any) -> None:
    calls: list[int] = []
    monkeypatch.setattr(runner_mod, "_seed_rng", calls.append)

    task = BenchTask(task_id="t", description="d", test_code="assert True\n")
    config = {"run_config": {"max_phase_iterations": 1}}  # no seed
    asyncio.run(run_benchmark([task], _FakeEngine("x = 1\n"), config))

    assert calls == []
