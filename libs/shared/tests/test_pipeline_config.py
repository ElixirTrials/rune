"""Tests for pipeline_config module."""

from __future__ import annotations

import json
from pathlib import Path

from shared.pipeline_config import (
    AdapterConfig,
    PipelineConfig,
    ReasoningLoopConfig,
    default_config,
    load_config,
    resolve_reasoning_loop_config,
)


def test_default_config_values() -> None:
    cfg = default_config()
    assert cfg.adapter.scaling == 0.075
    assert cfg.adapter.use_bias is True
    assert cfg.adapter.max_length == 2048
    assert cfg.generation.temperature == 0.3
    assert cfg.generation.repetition_penalty == 1.1
    assert cfg.prompt.style == "must_include"
    assert cfg.trajectory.style == "full_context"
    assert cfg.calibration.enabled is True
    assert cfg.calibration.n_trials == 5


def test_round_trip_json(tmp_path: Path) -> None:
    cfg = default_config()
    path = cfg.save(tmp_path / "test.json")
    loaded = load_config(path)
    assert loaded == cfg


def test_override_dotted_key() -> None:
    cfg = default_config()
    updated = cfg.override(**{"adapter.scaling": 0.1})
    assert updated.adapter.scaling == 0.1
    assert updated.generation.temperature == cfg.generation.temperature


def test_override_section_dict() -> None:
    cfg = default_config()
    updated = cfg.override(adapter={"scaling": 0.2, "use_bias": False})
    assert updated.adapter.scaling == 0.2
    assert updated.adapter.use_bias is False


def test_to_dict_and_back() -> None:
    cfg = default_config()
    d = cfg.to_dict()
    assert isinstance(d["calibration"]["scaling_range"], list)
    # Ensure JSON serializable
    json.dumps(d)


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nonexistent.json")
    assert cfg == default_config()


def test_partial_override_preserves_other_fields() -> None:
    cfg = default_config()
    updated = cfg.override(**{"generation.temperature": 0.7})
    assert updated.generation.temperature == 0.7
    assert updated.generation.max_tokens == cfg.generation.max_tokens
    assert updated.adapter == cfg.adapter


def test_adapter_config_frozen() -> None:
    ac = AdapterConfig()
    try:
        ac.scaling = 0.5  # type: ignore[misc]
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        pass


def test_config_is_frozen() -> None:
    cfg = PipelineConfig()
    try:
        cfg.adapter = AdapterConfig(scaling=0.5)  # type: ignore[misc]
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        pass


def test_decompose_config_defaults() -> None:
    cfg = default_config()
    assert hasattr(cfg, "decompose")
    assert cfg.decompose.skip_threshold == 200


def test_decompose_config_override() -> None:
    cfg = default_config()
    updated = cfg.override(**{"decompose.skip_threshold": 100})
    assert updated.decompose.skip_threshold == 100


def test_decompose_config_round_trip(tmp_path: Path) -> None:
    cfg = default_config()
    path = cfg.save(tmp_path / "test_decompose.json")
    loaded = load_config(path)
    assert loaded.decompose.skip_threshold == 200


def test_reasoning_loop_config_defaults():
    cfg = ReasoningLoopConfig()
    assert cfg.max_turns == 20
    assert cfg.context_budget_ratio == 0.75
    assert cfg.sliding_window_tokens == 1024
    assert cfg.chunk_threshold == 1024
    assert cfg.enable_chunk_composition is False
    assert cfg.code_scaling_boost == 1.2
    assert cfg.default_merge_method == "ties"
    assert cfg.collapse_cosine_threshold == 0.95
    assert cfg.collapse_norm_min == 0.1
    assert cfg.collapse_norm_max == 10.0
    assert cfg.collapse_repetition_threshold == 0.8


def test_reasoning_loop_config_phase_windows():
    cfg = ReasoningLoopConfig()
    assert cfg.phase_sliding_windows == {
        "decompose": 256,
        "plan": 512,
        "code": 1024,
        "code_repair": 1536,
        "integrate": 2048,
        "diagnose": 512,
    }


def test_pipeline_config_has_reasoning_loop():
    cfg = PipelineConfig()
    assert isinstance(cfg.reasoning_loop, ReasoningLoopConfig)
    assert cfg.reasoning_loop.max_turns == 20


def test_reasoning_loop_env_overrides(monkeypatch):
    monkeypatch.setenv("RUNE_MAX_REASONING_TURNS", "10")
    monkeypatch.setenv("RUNE_CONTEXT_BUDGET_RATIO", "0.5")
    monkeypatch.setenv("RUNE_SLIDING_WINDOW_TOKENS", "2048")
    monkeypatch.setenv("RUNE_ENABLE_CHUNK_COMPOSITION", "true")
    monkeypatch.setenv("RUNE_SLIDING_WINDOW_CODE", "512")
    cfg = resolve_reasoning_loop_config(ReasoningLoopConfig())
    assert cfg.max_turns == 10
    assert cfg.context_budget_ratio == 0.5
    assert cfg.sliding_window_tokens == 2048
    assert cfg.enable_chunk_composition is True
    assert cfg.phase_sliding_windows["code"] == 512


def test_reasoning_loop_round_trip(tmp_path: Path) -> None:
    cfg = default_config()
    path = cfg.save(tmp_path / "test_rl.json")
    loaded = load_config(path)
    assert loaded.reasoning_loop == cfg.reasoning_loop


