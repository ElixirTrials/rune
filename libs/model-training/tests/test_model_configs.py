"""Tests for model_training.model_configs module.

CPU-only tests — no GPU imports needed. Tests cover:
- ModelRegistry get/register/list operations
- Missing key raises KeyError with helpful message
- ModelConfig immutability (frozen)
- Built-in presets contain expected values
- validate_against_probe pass and mismatch cases
"""

from __future__ import annotations

from typing import Any

import pytest
from model_training.model_configs import (
    ModelConfig,
    ModelRegistry,
    validate_against_probe,
)

# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------


def test_registry_get_existing_preset() -> None:
    """Built-in presets resolve without error."""
    registry = ModelRegistry.default()
    config = registry.get("qwen3.5-9b")
    assert config.model_id == "Qwen/Qwen3.5-9B"
    assert config.warm_start_adapter_id == "danielcherubini/Qwen3.5-DeltaCoder-9B"


def test_registry_get_qwen3_coder_next() -> None:
    """qwen3-coder-next preset has expected values."""
    registry = ModelRegistry.default()
    config = registry.get("qwen3-coder-next")
    assert config.model_id == "Qwen/Qwen3-Coder-Next"
    assert config.warm_start_adapter_id is None
    assert config.default_lora_rank == 8


def test_registry_get_missing_raises_keyerror() -> None:
    """Missing preset raises KeyError with available names."""
    registry = ModelRegistry.default()
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_registry_list_names() -> None:
    """list_names returns sorted preset names."""
    registry = ModelRegistry.default()
    names = registry.list_names()
    assert "qwen3.5-9b" in names
    assert "qwen3-coder-next" in names
    assert names == sorted(names)


def test_registry_register_custom_config() -> None:
    """Custom configs can be registered and retrieved."""
    registry = ModelRegistry()
    custom = ModelConfig(
        canonical_name="test-model",
        model_id="test/model-123",
        expected_num_layers=24,
        expected_hidden_size=1024,
    )
    registry.register(custom)
    assert registry.get("test-model") is custom


def test_registry_default_is_singleton() -> None:
    """default() returns the same instance on repeated calls."""
    r1 = ModelRegistry.default()
    r2 = ModelRegistry.default()
    assert r1 is r2


# ---------------------------------------------------------------------------
# ModelConfig frozen
# ---------------------------------------------------------------------------


def test_model_config_frozen() -> None:
    """ModelConfig instances are immutable."""
    config = ModelConfig(
        canonical_name="frozen-test",
        model_id="test/frozen",
        expected_num_layers=32,
        expected_hidden_size=4096,
    )
    with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
        config.model_id = "something-else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Training defaults in presets
# ---------------------------------------------------------------------------


def test_qwen35_training_defaults() -> None:
    """qwen3.5-9b preset has strategy-aligned training defaults."""
    config = ModelRegistry.default().get("qwen3.5-9b")
    assert config.gradient_accumulation_steps == 16
    assert config.lr_scheduler_type == "constant"
    assert config.epochs == 3
    assert config.attn_implementation == "eager"
    assert config.default_lora_rank == 64
    assert config.default_lora_alpha == 32


def test_qwen3_coder_next_training_defaults() -> None:
    """qwen3-coder-next preset has its own training defaults."""
    config = ModelRegistry.default().get("qwen3-coder-next")
    assert config.gradient_accumulation_steps == 4
    assert config.lr_scheduler_type == "cosine"


# ---------------------------------------------------------------------------
# validate_against_probe
# ---------------------------------------------------------------------------


def test_validate_against_probe_passes() -> None:
    """No warnings when probe matches expected dimensions."""
    config = ModelConfig(
        canonical_name="valid-model",
        model_id="test/valid",
        expected_num_layers=32,
        expected_hidden_size=16,
    )
    probe: dict[str, Any] = {
        "attention_layer_indices": [3, 7, 11],
        "feature_sizes": {
            "q_proj": {"in": 16, "out": 32},
            "v_proj": {"in": 16, "out": 16},
        },
    }
    warnings = validate_against_probe(config, probe)
    assert warnings == []


def test_validate_against_probe_layer_mismatch() -> None:
    """Warning when probe finds layers beyond expected count."""
    config = ModelConfig(
        canonical_name="small-model",
        model_id="test/small",
        expected_num_layers=10,
        expected_hidden_size=16,
    )
    probe: dict[str, Any] = {
        "attention_layer_indices": [3, 7, 15],
        "feature_sizes": {},
    }
    warnings = validate_against_probe(config, probe)
    assert len(warnings) == 1
    assert "layer index 15" in warnings[0]


def test_validate_against_probe_hidden_size_mismatch() -> None:
    """Warning when q_proj input dim differs from expected hidden_size."""
    config = ModelConfig(
        canonical_name="dim-mismatch",
        model_id="test/dim",
        expected_num_layers=32,
        expected_hidden_size=1024,
    )
    probe: dict[str, Any] = {
        "attention_layer_indices": [0, 1],
        "feature_sizes": {
            "q_proj": {"in": 2048, "out": 2048},
        },
    }
    warnings = validate_against_probe(config, probe)
    assert len(warnings) == 1
    assert "q_proj" in warnings[0]
    assert "2048" in warnings[0]
