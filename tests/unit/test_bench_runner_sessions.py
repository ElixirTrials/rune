import asyncio
import json
from pathlib import Path

from rune.bench.runner import BenchTask, resolve_shipped_code, run_benchmark
from rune.engine.state import Feedback, StepRecord


class _FakeEngine:
    async def ainvoke(self, state, config):
        return {
            **state,
            "integrated_code": "print(1)",
            "code_results": {"_main": "print(1)"},
            "trajectory": [
                StepRecord(
                    step=0,
                    action_name="code",
                    target_subtask="_main",
                    adapter_id="a0",
                    feedback=Feedback(stdout="", stderr="", exit_code=0),
                    generated_code="print(1)",
                    trajectory_text="ROLE: coder",
                    prompt_text="p",
                    output_text="print(1)",
                )
            ],
        }


def test_run_benchmark_writes_session_with_verdict(tmp_path: Path) -> None:
    tasks = [BenchTask(task_id="t1", description="print 1", test_code="assert True")]
    config = {"run_config": {"max_phase_iterations": 3}, "benchmark": "mbpp"}
    asyncio.run(run_benchmark(tasks, _FakeEngine(), config, sessions_dir=tmp_path))
    assert (tmp_path / "t1" / "session.jsonl").exists()
    meta = json.loads((tmp_path / "t1" / "metadata.json").read_text())
    # write_session must run AFTER scoring, so the verdict is captured.
    assert meta["pass_at_1"] is True  # _FakeEngine emits code that passes `assert True`
    assert meta["schema_version"] == 2


class _NoShipEngine:
    async def ainvoke(self, state, config):
        return {
            **state,
            "integrated_code": "",
            "code_results": {},
            "best_code": {},
            "trajectory": [
                StepRecord(
                    step=0,
                    action_name="code",
                    target_subtask="f",
                    adapter_id="a0",
                    feedback=Feedback(stdout="", stderr="assert fail", exit_code=1),
                    generated_code="def f(): pass",
                    trajectory_text="ROLE: coder",
                    prompt_text="p",
                    output_text="def f(): pass",
                )
            ],
        }


def test_resolve_shipped_code_falls_back_to_best_attempt() -> None:
    task = BenchTask(
        task_id="t3",
        description="return 1",
        test_code="assert f() == 1",
        entry_point="f",
        public_checks="assert f() == 1",
    )
    wrong = "def f():\n    return 0\n"
    state = {
        "integrated_code": "",
        "code_results": {"f": wrong},
        "best_code": {"f": wrong},
        "best_quality": {"f": 2},
        "ship_best_on_exhaustion": True,
        "ship_best_min_quality": 1,
        "advisory_requirement_kinds": ("constraint_scale",),
        "complexity_probe_min_n": 8,
        "complexity_probe_max_n": 400,
        "complexity_probe_n_repeats": 3,
        "complexity_probe_per_run_timeout_s": 5.0,
    }
    assert resolve_shipped_code(state, task).strip() == wrong.strip()

    state["ship_best_on_exhaustion"] = False
    assert resolve_shipped_code(state, task) == ""


def test_run_benchmark_writes_session_when_nothing_ships(tmp_path: Path) -> None:
    tasks = [
        BenchTask(
            task_id="t2",
            description="noop",
            test_code="assert True",
            entry_point="f",
            public_checks="assert f() is None",
        )
    ]
    config = {"run_config": {"max_phase_iterations": 3}, "benchmark": "lcb"}
    asyncio.run(run_benchmark(tasks, _NoShipEngine(), config, sessions_dir=tmp_path))
    assert (tmp_path / "t2" / "session.jsonl").exists()
    meta = json.loads((tmp_path / "t2" / "metadata.json").read_text())
    assert meta["pass_at_1"] is False
    assert meta["schema_version"] == 2
