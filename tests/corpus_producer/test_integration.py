"""End-to-end integration test for the phase corpus producer.

Uses:
  - A fake pipeline runner injected via monkeypatching
  - Mocked run_benchmark returning Pass@1=1.0
  - Real ProgressDB, bin_key(), emit_bin_manifest, invoke_bin_training(dry_run=True)

Verifies the full produce_corpus() flow produces the expected manifest files
and bin record counts without touching GPU or the real benchmark harness.

Note: produce_corpus() lives in scripts/phase_corpus_producer.py and is
imported here by adding scripts/ to sys.path (same mechanism as bootstrap.py).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make scripts/ importable so we can import phase_corpus_producer
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import PipelineRunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BENCHMARK = "humaneval"
_PROBLEM_ID = "HumanEval/0"
_PHASES = ["decompose", "plan", "code", "integrate"]


def _make_artifacts(
    benchmark: str = _BENCHMARK, problem_id: str = _PROBLEM_ID
) -> list[PhaseArtifact]:
    return [
        PhaseArtifact(
            phase=p,
            benchmark=benchmark,
            problem_id=problem_id,
            pipeline_run_id="run-test",
            input_text=f"input for {p}",
            output_text=f"output for {p}",
        )
        for p in _PHASES
    ]


def _fake_pipeline_runner(
    benchmark: str,
    problem_id: str,
    prompt: str,
    *,
    timeout: int = 300,
    base_model_id: str = "Qwen/Qwen3.5-9B",
) -> PipelineRunResult:
    return PipelineRunResult(
        run_id="run-test",
        benchmark=benchmark,
        problem_id=problem_id,
        artifacts=_make_artifacts(benchmark, problem_id),
        final_code="def solution(): pass",
        success=True,
    )


def _mock_run_benchmark_pass(
    model_adapter_stack: object,
    benchmark_id: str,
    problem_ids: list[str] | None = None,
    max_samples: int = 1,
) -> MagicMock:
    verdict = MagicMock()
    verdict.passed = True
    result = MagicMock()
    result.per_problem = {pid: verdict for pid in (problem_ids or [_PROBLEM_ID])}
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
def test_produce_corpus_one_problem_emits_manifests(mock_rb: MagicMock) -> None:
    """One HumanEval problem, Pass@1=1.0 → 4 manifest files (one per phase)."""
    import phase_corpus_producer as pcp  # noqa: PLC0415

    with patch.object(pcp, "run_pipeline_for_problem", side_effect=_fake_pipeline_runner):
        with patch.object(
            pcp, "_load_problems", return_value=[(_PROBLEM_ID, "Sort a list of integers.")]
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                counts = pcp.produce_corpus(
                    benchmarks=[_BENCHMARK],
                    out_dir=Path(tmpdir),
                    skip_training=True,
                )

    assert "decompose_humaneval" in counts
    assert "plan_humaneval" in counts
    assert "code_humaneval" in counts
    assert "integrate_humaneval" in counts
    assert "diagnose_pooled" not in counts  # no diagnose phase in this run


@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
def test_produce_corpus_dry_run_does_not_train(mock_rb: MagicMock) -> None:
    """dry_run=True should emit manifests but not call real train_and_register."""
    import phase_corpus_producer as pcp  # noqa: PLC0415

    with patch.object(pcp, "run_pipeline_for_problem", side_effect=_fake_pipeline_runner):
        with patch.object(
            pcp, "_load_problems", return_value=[(_PROBLEM_ID, "Sort a list.")]
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("corpus_producer.trainer_bridge.train_and_register") as mock_train:
                    pcp.produce_corpus(
                        benchmarks=[_BENCHMARK],
                        out_dir=Path(tmpdir),
                        dry_run=True,
                    )
                    # dry_run=True means invoke_bin_training(dry_run=True) returns early
                    mock_train.assert_not_called()


@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
def test_produce_corpus_resume_skips_done_problems(mock_rb: MagicMock) -> None:
    """Second run with same out_dir skips already-done problem."""
    import phase_corpus_producer as pcp  # noqa: PLC0415

    call_count: list[int] = [0]

    def counting_runner(bm: str, pid: str, prompt: str, **kw: object) -> PipelineRunResult:
        call_count[0] += 1
        return _fake_pipeline_runner(bm, pid, prompt, **kw)  # type: ignore[arg-type]

    with patch.object(pcp, "run_pipeline_for_problem", side_effect=counting_runner):
        with patch.object(
            pcp, "_load_problems", return_value=[(_PROBLEM_ID, "prompt")]
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                # First run
                pcp.produce_corpus(
                    benchmarks=[_BENCHMARK],
                    out_dir=Path(tmpdir),
                    skip_training=True,
                )
                first_count = call_count[0]

                # Second run (same out_dir, force=False)
                call_count[0] = 0
                pcp.produce_corpus(
                    benchmarks=[_BENCHMARK],
                    out_dir=Path(tmpdir),
                    skip_training=True,
                )
                second_count = call_count[0]

    assert first_count >= 1
    assert second_count == 0, "Resume should skip already-done problems"


@patch("corpus_producer.success_filter.run_benchmark", side_effect=_mock_run_benchmark_pass)
def test_produce_corpus_force_reruns_done_problems(mock_rb: MagicMock) -> None:
    """--force re-runs even done problems."""
    import phase_corpus_producer as pcp  # noqa: PLC0415

    call_count: list[int] = [0]

    def counting_runner(bm: str, pid: str, prompt: str, **kw: object) -> PipelineRunResult:
        call_count[0] += 1
        return _fake_pipeline_runner(bm, pid, prompt, **kw)  # type: ignore[arg-type]

    with patch.object(pcp, "run_pipeline_for_problem", side_effect=counting_runner):
        with patch.object(
            pcp, "_load_problems", return_value=[(_PROBLEM_ID, "prompt")]
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                pcp.produce_corpus(
                    benchmarks=[_BENCHMARK],
                    out_dir=Path(tmpdir),
                    skip_training=True,
                )
                call_count[0] = 0
                pcp.produce_corpus(
                    benchmarks=[_BENCHMARK],
                    out_dir=Path(tmpdir),
                    skip_training=True,
                    force=True,
                )
                assert call_count[0] > 0
