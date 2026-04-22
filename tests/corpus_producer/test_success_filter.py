"""Tests for corpus_producer.success_filter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from corpus_producer.models import PhaseArtifact
from corpus_producer.success_filter import filter_artifacts


def _arts(phases: list[str], benchmark: str = "humaneval") -> list[PhaseArtifact]:
    return [
        PhaseArtifact(
            phase=p,
            benchmark=benchmark,
            problem_id="HE/0",
            pipeline_run_id="run-1",
            input_text="prompt",
            output_text=f"output for {p}",
        )
        for p in phases
    ]


def _mock_verdict(passed: bool) -> MagicMock:
    verdict = MagicMock()
    verdict.passed = passed
    result = MagicMock()
    result.per_problem = {"HE/0": verdict}
    return result


@patch("corpus_producer.success_filter.run_benchmark")
def test_pass_keeps_all_phases(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(True)
    arts = _arts(["decompose", "plan", "code", "integrate"])
    kept = filter_artifacts(arts, "def f(): pass", "humaneval", "HE/0")
    assert len(kept) == 4
    assert all(a.pass_at_1 is True for a in kept)


@patch("corpus_producer.success_filter.run_benchmark")
def test_fail_drops_non_diagnose(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(False)
    arts = _arts(["decompose", "plan", "code", "integrate"])
    kept = filter_artifacts(arts, "", "humaneval", "HE/0")
    assert len(kept) == 0


@patch("corpus_producer.success_filter.run_benchmark")
def test_fail_keeps_diagnose_when_repair_succeeded(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(False)
    arts = _arts(["decompose", "diagnose"])
    arts[1].metadata["repair_succeeded"] = "true"
    kept = filter_artifacts(arts, "", "humaneval", "HE/0")
    assert len(kept) == 1
    assert kept[0].phase == "diagnose"


@patch("corpus_producer.success_filter.run_benchmark")
def test_fail_drops_diagnose_when_repair_not_succeeded(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(False)
    arts = _arts(["diagnose"])
    # no repair_succeeded metadata
    kept = filter_artifacts(arts, "", "humaneval", "HE/0")
    assert len(kept) == 0


@patch("corpus_producer.success_filter.run_benchmark")
def test_run_benchmark_called_with_correct_args(mock_rb: MagicMock) -> None:
    mock_rb.return_value = _mock_verdict(True)
    arts = _arts(["integrate"])
    filter_artifacts(arts, "code", "mbpp", "MBPP/5")
    mock_rb.assert_called_once()
    kwargs = mock_rb.call_args
    assert kwargs[1]["benchmark_id"] == "mbpp" or kwargs[0][1] == "mbpp"
