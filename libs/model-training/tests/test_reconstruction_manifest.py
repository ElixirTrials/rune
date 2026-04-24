"""Tests for ReconstructionRecord / ReconstructionManifest round-trip + validation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from model_training.reconstruction.manifest import (
    ReconstructionManifest,
    ReconstructionRecord,
    validate_homogeneity,
)


def _rec(**overrides: object) -> ReconstructionRecord:
    defaults: dict[str, object] = {
        "task_id": "adapter-001",
        "adapter_path": "/adapters/adapter-001",
        "task_description": "fix off-by-one in list slicing",
        "base_model_id": "Qwen/Qwen3.5-9B",
        "warm_start_adapter": "danielcherubini/Qwen3.5-DeltaCoder-9B",
        "rank": 64,
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "layer_indices": tuple(range(32)),
        "created_at": "2026-04-22T00:00:00Z",
        "source_task_hash": "task-hash-001",
        "fitness_score": 0.82,
    }
    defaults.update(overrides)
    return ReconstructionRecord(**defaults)  # type: ignore[arg-type]


def test_record_roundtrips_via_to_dict_from_dict() -> None:
    original = _rec()
    round_tripped = ReconstructionRecord.from_dict(original.to_dict())
    assert round_tripped == original


def test_manifest_roundtrips_via_json(tmp_path: Path) -> None:
    manifest = ReconstructionManifest(
        schema_version=1,
        base_model_id="Qwen/Qwen3.5-9B",
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        rank=64,
        layer_indices=tuple(range(32)),
        task_embedding_model="sentence-transformers/all-mpnet-base-v2",
        task_embedding_dim=768,
        records=[_rec(task_id="a"), _rec(task_id="b")],
        zscore_stats_path=None,
        created_at="2026-04-22T00:00:00Z",
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ReconstructionManifest.load(path)
    assert loaded == manifest
    # Must be human-readable JSON.
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
    assert len(parsed["records"]) == 2


def test_validate_homogeneity_passes_for_matching_records() -> None:
    records = [_rec(task_id="a"), _rec(task_id="b")]
    validate_homogeneity(
        records,
        base_model_id="Qwen/Qwen3.5-9B",
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        rank=64,
        layer_indices=tuple(range(32)),
    )  # no raise


def test_validate_homogeneity_raises_on_base_mismatch() -> None:
    records = [
        _rec(task_id="a"),
        _rec(task_id="b", base_model_id="Qwen/Qwen2.5-Coder-7B"),
    ]
    with pytest.raises(ValueError, match="base_model_id"):
        validate_homogeneity(
            records,
            base_model_id="Qwen/Qwen3.5-9B",
            warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=64,
            layer_indices=tuple(range(32)),
        )


def test_validate_homogeneity_raises_on_rank_mismatch() -> None:
    records = [_rec(task_id="a"), _rec(task_id="b", rank=32)]
    with pytest.raises(ValueError, match="rank"):
        validate_homogeneity(
            records,
            base_model_id="Qwen/Qwen3.5-9B",
            warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=64,
            layer_indices=tuple(range(32)),
        )


def test_validate_homogeneity_raises_on_target_modules_mismatch() -> None:
    records = [
        _rec(task_id="a"),
        _rec(task_id="b", target_modules=("q_proj", "v_proj")),
    ]
    with pytest.raises(ValueError, match="target_modules"):
        validate_homogeneity(
            records,
            base_model_id="Qwen/Qwen3.5-9B",
            warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=64,
            layer_indices=tuple(range(32)),
        )


def test_manifest_module_is_cpu_importable() -> None:
    # No torch / safetensors at import time.
    mod = importlib.import_module("model_training.reconstruction.manifest")
    assert hasattr(mod, "ReconstructionRecord")
