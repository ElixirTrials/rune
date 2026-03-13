"""Failing tests for model_training.d2l_config module.

Tests cover:
- get_d2l_qwen3_config returns correct Qwen3-Coder-Next dimensions
- build_qwen3_hypernet_config produces a HypernetConfig with 12 attention layer indices
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# get_d2l_qwen3_config tests
# ---------------------------------------------------------------------------


def test_get_d2l_qwen3_config_returns_required_keys() -> None:
    """get_d2l_qwen3_config returns dict with all required architecture keys."""
    from model_training.d2l_config import get_d2l_qwen3_config

    config = get_d2l_qwen3_config()
    required_keys = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "attention_layer_indices",
        "vocab_size",
        "model_type",
    }
    assert required_keys.issubset(config.keys()), (
        f"Missing keys: {required_keys - config.keys()}"
    )


def test_get_d2l_qwen3_config_correct_dimensions() -> None:
    """get_d2l_qwen3_config returns correct Qwen3-Coder-Next dimensions."""
    from model_training.d2l_config import get_d2l_qwen3_config

    config = get_d2l_qwen3_config()
    assert config["hidden_size"] == 2048
    assert config["num_hidden_layers"] == 48
    assert len(config["attention_layer_indices"]) == 12


def test_get_d2l_qwen3_config_attention_layer_indices() -> None:
    """get_d2l_qwen3_config returns the exact 12 full_attention layer indices."""
    from model_training.d2l_config import get_d2l_qwen3_config

    config = get_d2l_qwen3_config()
    expected_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
    assert config["attention_layer_indices"] == expected_indices


# ---------------------------------------------------------------------------
# build_qwen3_hypernet_config tests
# ---------------------------------------------------------------------------


def test_build_qwen3_hypernet_config_returns_twelve_layer_indices() -> None:
    """build_qwen3_hypernet_config returns object with exactly 12 layer_indices."""
    from model_training.d2l_config import build_qwen3_hypernet_config

    hypernet_cfg = build_qwen3_hypernet_config()
    layer_indices = list(hypernet_cfg.layer_indices)
    assert len(layer_indices) == 12, (
        f"Expected 12 layer indices, got {len(layer_indices)}"
    )


def test_build_qwen3_hypernet_config_layer_indices_match_qwen3_config() -> None:
    """build_qwen3_hypernet_config layer_indices matches get_d2l_qwen3_config."""
    from model_training.d2l_config import (  # noqa: PLC0415
        build_qwen3_hypernet_config,
        get_d2l_qwen3_config,
    )

    config = get_d2l_qwen3_config()
    hypernet_cfg = build_qwen3_hypernet_config()
    assert list(hypernet_cfg.layer_indices) == config["attention_layer_indices"]


def test_build_qwen3_hypernet_config_base_hidden_size() -> None:
    """build_qwen3_hypernet_config sets base_hidden_size to 2048."""
    from model_training.d2l_config import build_qwen3_hypernet_config

    hypernet_cfg = build_qwen3_hypernet_config()
    assert hypernet_cfg.base_hidden_size == 2048
