"""Tests for corpus_producer.rationalization."""

from __future__ import annotations

from collections.abc import Callable

from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import PipelineRunResult
from corpus_producer.rationalization import (
    MIN_EXAMPLES_PER_BIN,
    _build_hint_prompt,
    star_rationalize,
)


def _art(phase: str = "decompose", rationalized: bool = False) -> PhaseArtifact:
    a = PhaseArtifact(
        phase=phase,
        benchmark="humaneval",
        problem_id="HE/0",
        pipeline_run_id="r",
        input_text="in",
        output_text="out",
        pass_at_1=True,
    )
    a.rationalized = rationalized
    return a


def _make_runner(
    success: bool, artifacts: list[PhaseArtifact]
) -> Callable[..., PipelineRunResult]:
    """Return a callable that mimics PipelineRunnerProtocol."""

    def runner(
        benchmark: str, problem_id: str, prompt: str, **kw: object
    ) -> PipelineRunResult:
        return PipelineRunResult(
            run_id="r2",
            benchmark=benchmark,
            problem_id=problem_id,
            artifacts=artifacts,
            final_code="def f(): pass",
            success=success,
        )

    return runner


def _filter_all_pass(
    arts: list[PhaseArtifact], code: str, bm: str, pid: str
) -> list[PhaseArtifact]:
    for a in arts:
        a.pass_at_1 = True
    return arts


def _filter_none_pass(
    arts: list[PhaseArtifact], code: str, bm: str, pid: str
) -> list[PhaseArtifact]:
    return []


def test_build_hint_prompt_includes_hints() -> None:
    result = _build_hint_prompt("Sort a list.", ["test_sort: PASS"])
    assert "test_sort: PASS" in result
    assert "Sort a list." in result


def test_rationalize_marks_artifacts_rationalized() -> None:
    failing = [("humaneval", "HE/1", "prompt")]
    hints = {"HE/1": ["test_a: FAIL"]}
    runner = _make_runner(True, [_art()])
    new = star_rationalize(
        "decompose_humaneval",
        existing_artifacts=[],
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=runner,
        success_filter_fn=_filter_all_pass,
    )
    assert all(a.rationalized is True for a in new)


def test_rationalize_stops_when_bin_full() -> None:
    # existing_artifacts already at MIN - 1; one rationalization should fill it
    existing = [_art() for _ in range(MIN_EXAMPLES_PER_BIN - 1)]
    failing = [("humaneval", f"HE/{i}", "p") for i in range(10)]
    hints = {f"HE/{i}": ["test: PASS"] for i in range(10)}

    calls: list[str] = []

    def counting_runner(
        bm: str, pid: str, prompt: str, **kw: object
    ) -> PipelineRunResult:
        calls.append(pid)
        return PipelineRunResult(
            run_id="r",
            benchmark=bm,
            problem_id=pid,
            artifacts=[_art()],
            final_code="",
            success=True,
        )

    star_rationalize(
        "decompose_humaneval",
        existing_artifacts=existing,
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=counting_runner,
        success_filter_fn=_filter_all_pass,
    )
    # Should stop after 1 call (existing 59 + 1 new = 60)
    assert len(calls) == 1


def test_rationalize_skips_problem_with_no_hints() -> None:
    failing = [("humaneval", "HE/no_hint", "prompt")]
    hints: dict[str, list[str]] = {}

    calls: list[str] = []

    def counting_runner(
        bm: str, pid: str, prompt: str, **kw: object
    ) -> PipelineRunResult:
        calls.append(pid)
        return PipelineRunResult(
            run_id="r", benchmark=bm, problem_id=pid, success=False
        )

    new = star_rationalize(
        "decompose_humaneval",
        existing_artifacts=[],
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=counting_runner,
        success_filter_fn=_filter_all_pass,
    )
    assert calls == []
    assert new == []


def test_rationalize_returns_empty_when_filter_rejects() -> None:
    failing = [("humaneval", "HE/1", "p")]
    hints = {"HE/1": ["test: FAIL"]}
    runner = _make_runner(True, [_art()])
    new = star_rationalize(
        "decompose_humaneval",
        existing_artifacts=[],
        failing_problems=failing,
        test_hints_by_problem=hints,
        pipeline_runner=runner,
        success_filter_fn=_filter_none_pass,
    )
    assert new == []
