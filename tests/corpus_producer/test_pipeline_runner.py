"""Tests for corpus_producer.pipeline_runner."""

from __future__ import annotations

from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import (
    PipelineRunResult,
    PipelineRunnerProtocol,
    _parse_artifacts_from_output,
    run_pipeline_for_problem,
)


def _fake_output(with_diagnose: bool = False) -> dict[str, object]:
    out: dict[str, object] = {
        "project_prompt": "Write a sort function.",
        "final_code": "def solution(): pass",
        "phase_results": {
            "decompose": {"output": "1. parse — parse input"},
            "plan": {"plans": {"parse": "Use argparse"}},
            "code": {"outputs": {"parse": "def parse(): pass"}},
            "integrate": {"generated_code": "def solution(): pass", "stderr": ""},
        },
    }
    if with_diagnose:
        out["phase_results"]["repair"] = {"diagnosis": "NameError on line 3"}  # type: ignore[index]
    return out


def test_parse_artifacts_returns_four_phases():
    arts = _parse_artifacts_from_output("run-1", "humaneval", "HumanEval/0", _fake_output())
    phases = [a.phase for a in arts]
    assert "decompose" in phases
    assert "plan" in phases
    assert "code" in phases
    assert "integrate" in phases


def test_parse_artifacts_diagnose_present_when_repair_fired():
    arts = _parse_artifacts_from_output("run-1", "humaneval", "HumanEval/0", _fake_output(with_diagnose=True))
    phases = [a.phase for a in arts]
    assert "diagnose" in phases


def test_parse_artifacts_no_diagnose_when_no_repair():
    arts = _parse_artifacts_from_output("run-1", "humaneval", "HumanEval/0", _fake_output())
    phases = [a.phase for a in arts]
    assert "diagnose" not in phases


def test_parse_artifacts_benchmark_and_problem_tagged():
    arts = _parse_artifacts_from_output("run-1", "mbpp", "MBPP/42", _fake_output())
    for art in arts:
        assert art.benchmark == "mbpp"
        assert art.problem_id == "MBPP/42"


def test_parse_artifacts_pipeline_run_id_set():
    arts = _parse_artifacts_from_output("run-xyz", "humaneval", "HumanEval/0", _fake_output())
    for art in arts:
        assert art.pipeline_run_id == "run-xyz"


class FakeRunner:
    """Test double satisfying PipelineRunnerProtocol."""

    def __init__(self, result: PipelineRunResult) -> None:
        self._result = result

    def __call__(
        self,
        benchmark: str,
        problem_id: str,
        problem_prompt: str,
        *,
        timeout: int = 300,
        base_model_id: str = "Qwen/Qwen3.5-9B",
    ) -> PipelineRunResult:
        return self._result


def test_fake_runner_satisfies_protocol():
    result = PipelineRunResult(run_id="r", benchmark="humaneval", problem_id="HE/0")
    runner = FakeRunner(result)
    assert isinstance(runner, PipelineRunnerProtocol)


def test_pipeline_run_result_defaults():
    r = PipelineRunResult(run_id="r", benchmark="b", problem_id="p")
    assert r.success is False
    assert r.artifacts == []
    assert r.final_code == ""
