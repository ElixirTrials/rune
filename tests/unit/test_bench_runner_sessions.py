import asyncio
import json
from pathlib import Path

from rune.bench.runner import BenchTask, run_benchmark
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
