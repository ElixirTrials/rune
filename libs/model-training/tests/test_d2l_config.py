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


# ---------------------------------------------------------------------------
# build_hypernet_config tests (model-agnostic)
# ---------------------------------------------------------------------------


def test_build_hypernet_config_delegates_qwen3_coder_next() -> None:
    """build_hypernet_config("qwen3-coder-next") produces same result as direct."""
    from model_training.d2l_config import (
        build_hypernet_config,
        build_qwen3_hypernet_config,
    )

    direct = build_qwen3_hypernet_config()
    via_registry = build_hypernet_config("qwen3-coder-next")
    assert list(via_registry.layer_indices) == list(direct.layer_indices)
    assert via_registry.base_hidden_size == direct.base_hidden_size


def test_build_hypernet_config_qwen35_with_probe_cache(
    tmp_path: object,
    monkeypatch: object,
) -> None:
    """build_hypernet_config("qwen3.5-9b") builds config from probe cache."""
    import model_training.d2l_probe as probe_module
    from model_training.d2l_probe import save_probe_cache

    # Use tmp_path for probe cache
    monkeypatch.setattr(probe_module, "PROBE_CACHE_DIR", tmp_path)  # type: ignore[arg-type]

    # Populate a fake probe cache for qwen3.5-9b.
    # Dimensions reflect the registry entry for Qwen3.5-9B (32 layers,
    # hidden_size=4096). v_proj out projects to a smaller dim under GQA.
    probe_data = {
        "attention_layer_indices": list(range(32)),
        "feature_sizes": {
            "q_proj": {"in": 4096, "out": 4096},
            "v_proj": {"in": 4096, "out": 512},
        },
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    save_probe_cache("qwen3.5-9b", probe_data)

    from model_training.d2l_config import build_hypernet_config

    hc = build_hypernet_config("qwen3.5-9b")
    assert list(hc.layer_indices) == list(range(32))
    assert hc.base_hidden_size == 4096


def test_build_hypernet_config_unknown_model_raises() -> None:
    """build_hypernet_config raises KeyError for unregistered model."""
    import pytest
    from model_training.d2l_config import build_hypernet_config

    with pytest.raises(KeyError, match="nonexistent-model"):
        build_hypernet_config("nonexistent-model")


def test_build_hypernet_config_missing_probe_raises(
    tmp_path: object,
    monkeypatch: object,
) -> None:
    """build_hypernet_config raises RuntimeError when no probe cache exists."""
    import model_training.d2l_probe as probe_module
    import pytest

    monkeypatch.setattr(probe_module, "PROBE_CACHE_DIR", tmp_path)  # type: ignore[arg-type]

    from model_training.d2l_config import build_hypernet_config

    with pytest.raises(RuntimeError, match="probe cache"):
        build_hypernet_config("qwen3.5-9b")
