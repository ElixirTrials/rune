"""Tests for corpus_producer.models.PhaseArtifact."""

import pytest
from corpus_producer.models import PhaseArtifact


def _make(phase: str = "decompose", benchmark: str = "humaneval") -> PhaseArtifact:
    return PhaseArtifact(
        phase=phase,
        benchmark=benchmark,
        problem_id="HumanEval/0",
        pipeline_run_id="run-abc",
        input_text="## Task\nWrite a function.",
        output_text="1. subtask_a — parse input",
        pass_at_1=True,
    )


def test_bin_key_non_diagnose():
    art = _make(phase="decompose", benchmark="humaneval")
    assert art.bin_key() == "decompose_humaneval"


def test_bin_key_diagnose_always_pooled():
    art = _make(phase="diagnose", benchmark="mbpp")
    assert art.bin_key() == "diagnose_pooled"


def test_to_manifest_record_keys():
    art = _make()
    rec = art.to_manifest_record()
    assert "task_id" in rec
    assert "activation_text" in rec
    assert "teacher_text" in rec
    assert "metadata" in rec


def test_to_manifest_record_task_id_format():
    art = _make(phase="plan", benchmark="mbpp")
    art.problem_id = "MBPP/1"
    rec = art.to_manifest_record()
    assert rec["task_id"] == "mbpp/MBPP/1/plan"


def test_to_manifest_record_teacher_contains_output():
    art = _make()
    rec = art.to_manifest_record()
    assert art.output_text in rec["teacher_text"]


def test_to_manifest_record_metadata_provenance():
    art = _make(phase="code", benchmark="apps")
    art.rationalized = True
    rec = art.to_manifest_record()
    assert rec["metadata"]["rationalized"] is True
    assert rec["metadata"]["phase"] == "code"


def test_rationalized_defaults_false():
    art = _make()
    assert art.rationalized is False


def test_metadata_extra_fields_forwarded():
    art = _make()
    art.metadata["subtask"] = "parse_input"
    rec = art.to_manifest_record()
    assert rec["metadata"]["subtask"] == "parse_input"
