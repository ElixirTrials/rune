"""Tests for corpus_producer.binning."""

from corpus_producer.binning import DIAGNOSE_BIN_KEY, bin_artifacts, expected_bin_keys
from corpus_producer.models import PhaseArtifact


def _art(phase: str, benchmark: str = "humaneval") -> PhaseArtifact:
    return PhaseArtifact(
        phase=phase,
        benchmark=benchmark,
        problem_id="P/0",
        pipeline_run_id="r",
        input_text="in",
        output_text="out",
    )


def test_bin_artifacts_groups_by_phase_benchmark():
    arts = [_art("decompose", "humaneval"), _art("plan", "humaneval"), _art("decompose", "mbpp")]
    bins = bin_artifacts(arts)
    assert "decompose_humaneval" in bins
    assert "plan_humaneval" in bins
    assert "decompose_mbpp" in bins
    assert len(bins["decompose_humaneval"]) == 1


def test_bin_artifacts_diagnose_pooled():
    arts = [_art("diagnose", "humaneval"), _art("diagnose", "mbpp")]
    bins = bin_artifacts(arts)
    assert DIAGNOSE_BIN_KEY in bins
    assert len(bins[DIAGNOSE_BIN_KEY]) == 2
    assert "diagnose_humaneval" not in bins


def test_bin_artifacts_empty_list():
    assert bin_artifacts([]) == {}


def test_expected_bin_keys_count():
    keys = expected_bin_keys()
    assert len(keys) == 25  # 4 phases × 6 benchmarks + 1 diagnose_pooled


def test_expected_bin_keys_contains_diagnose_pooled():
    assert DIAGNOSE_BIN_KEY in expected_bin_keys()


def test_expected_bin_keys_custom_benchmarks():
    keys = expected_bin_keys(["humaneval", "mbpp"])
    # 4 phases × 2 benchmarks + 1 diagnose_pooled = 9
    assert len(keys) == 9
