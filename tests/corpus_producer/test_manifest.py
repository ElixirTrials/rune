"""Tests for corpus_producer.manifest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from corpus_producer.manifest import emit_bin_manifest, load_bin_manifest
from corpus_producer.models import PhaseArtifact


def _art(
    phase: str = "decompose",
    benchmark: str = "humaneval",
    pid: str = "HE/0",
) -> PhaseArtifact:
    return PhaseArtifact(
        phase=phase,
        benchmark=benchmark,
        problem_id=pid,
        pipeline_run_id="run-1",
        input_text="## Task\nSort a list.",
        output_text="1. parse — read input",
        pass_at_1=True,
    )


def test_emit_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = emit_bin_manifest("decompose_humaneval", [_art()], tmpdir)
        assert p.exists()
        assert p.name == "decompose_humaneval.jsonl"


def test_emit_correct_line_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = [_art(pid=f"HE/{i}") for i in range(5)]
        p = emit_bin_manifest("decompose_humaneval", arts, tmpdir)
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == 5


def test_emit_record_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = emit_bin_manifest("decompose_humaneval", [_art()], tmpdir)
        rec = json.loads(p.read_text().splitlines()[0])
        assert "task_id" in rec
        assert "activation_text" in rec
        assert "teacher_text" in rec
        assert "metadata" in rec


def test_emit_raises_on_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="empty bin"):
            emit_bin_manifest("decompose_humaneval", [], tmpdir)


def test_emit_diagnose_pooled_bin_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        art = _art(phase="diagnose")
        p = emit_bin_manifest("diagnose_pooled", [art], tmpdir)
        assert p.name == "diagnose_pooled.jsonl"


def test_load_bin_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        arts = [_art(pid=f"HE/{i}") for i in range(3)]
        p = emit_bin_manifest("decompose_humaneval", arts, tmpdir)
        records = load_bin_manifest(p)
        assert len(records) == 3
        assert all("task_id" in r for r in records)


def test_emit_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "c"
        p = emit_bin_manifest(
            "plan_mbpp", [_art(phase="plan", benchmark="mbpp")], nested
        )
        assert p.exists()
