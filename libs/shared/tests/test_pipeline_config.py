"""Tests for pipeline_config module."""

from __future__ import annotations

import json
from pathlib import Path

from shared.pipeline_config import (
    AdapterConfig,
    PipelineConfig,
    default_config,
    load_config,
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
