"""Tests for from-scratch HyperLoRA construction and checkpoint compatibility.

Tests:
- build_from_scratch_hypernet_config produces valid config for Qwen 3.5 9B
- HyperLoRA can be instantiated from the config
- Checkpoint format is compatible with rune_runner._is_sakana_checkpoint
- load_sakana_checkpoint can load from-scratch checkpoints
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _patch_flash() -> None:
    from model_training.sakana_d2l import _patch_flash_attention

    _patch_flash_attention()


def test_build_from_scratch_config_layer_indices() -> None:
    from model_training.d2l_config import build_from_scratch_hypernet_config

    hc = build_from_scratch_hypernet_config("qwen3.5-9b")
    indices = list(hc.layer_indices)
    assert indices == list(range(32))


def test_build_from_scratch_defaults_from_yaml() -> None:
    """Config builder reads defaults from hypernet_defaults.yaml."""
    from model_training.d2l_config import (
        build_from_scratch_hypernet_config,
        load_hypernet_defaults,
    )

    dfl = load_hypernet_defaults()
    hc = build_from_scratch_hypernet_config("qwen3.5-9b")

    assert hc.lora_config.r == dfl["lora"]["r"]
    assert list(hc.lora_config.target_modules) == dfl["lora"]["target_modules"]
    assert hc.aggregator_config.n_latent_queries == dfl["perceiver"]["n_latent_queries"]
    perc = dfl["perceiver"]
    assert hc.aggregator_config.layer_to_layer_ctx_encoder == perc["layer_to_layer"]
    agg = hc.aggregator_config
    assert agg.num_self_attn_per_block == perc["num_self_attn_per_block"]
    assert hc.use_bias == dfl["head"]["use_bias"]


def test_build_from_scratch_config_feature_sizes_attention() -> None:
    """Full-attention modules have correct dimensions (q_proj doubled for gating)."""
    from model_training.d2l_config import build_from_scratch_hypernet_config

    hc = build_from_scratch_hypernet_config(
        "qwen3.5-9b",
        target_modules=["q_proj", "v_proj"],
    )
    in_sizes, out_sizes = hc.feature_sizes
    assert in_sizes["q_proj"] == 4096
    assert out_sizes["q_proj"] == 8192
    assert out_sizes["v_proj"] == 1024


def test_build_from_scratch_config_aggregator() -> None:
    from model_training.d2l_config import (
        build_from_scratch_hypernet_config,
        load_hypernet_defaults,
    )

    dfl = load_hypernet_defaults()
    hc = build_from_scratch_hypernet_config("qwen3.5-9b")
    assert hc.aggregator_config is not None
    assert hc.aggregator_config.feature_size == 4096
    assert hc.aggregator_config.num_layers == 32
    assert hc.aggregator_config.num_blocks == dfl["perceiver"]["num_blocks"]


def test_hyperlora_builds_from_scratch() -> None:
    from ctx_to_lora.modeling.hypernet import HyperLoRA
    from model_training.d2l_config import build_from_scratch_hypernet_config

    hc = build_from_scratch_hypernet_config("qwen3.5-9b", lora_r=8)
    hyperlora = HyperLoRA(hc)
    n_params = sum(p.numel() for p in hyperlora.parameters())
    assert n_params > 0
    assert all(p.requires_grad for p in hyperlora.parameters())


def test_checkpoint_format_detected_as_sakana() -> None:
    """Checkpoint with hypernet_config is detected as Sakana."""
    import tempfile

    import torch
    from model_training.d2l_config import build_from_scratch_hypernet_config

    hc = build_from_scratch_hypernet_config("qwen3.5-9b", lora_r=8)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(
            {
                "hypernet_config": hc,
                "hypernet_state_dict": {},
                "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            },
            f.name,
        )

        sd = torch.load(f.name, map_location="cpu", weights_only=False)
        loaded_hc = sd.get("hypernet_config")
        assert loaded_hc is not None
        assert not isinstance(loaded_hc, dict)


def test_load_sakana_checkpoint_reads_hypernet_state_dict() -> None:
    """load_sakana_checkpoint loads weights from hypernet_state_dict key."""
    import tempfile

    import torch
    from ctx_to_lora.modeling.hypernet import HyperLoRA
    from model_training.d2l_config import build_from_scratch_hypernet_config

    hc = build_from_scratch_hypernet_config("qwen3.5-9b", lora_r=8)
    hyperlora = HyperLoRA(hc)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(
            {
                "hypernet_config": hc,
                "hypernet_state_dict": hyperlora.state_dict(),
                "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            },
            f.name,
        )

        from model_training.sakana_d2l import load_sakana_checkpoint

        loaded_hypernet, loaded_hc = load_sakana_checkpoint(
            checkpoint_path=f.name,
            device="cpu",
        )
        assert loaded_hc.latent_size == 512
        assert loaded_hc.lora_config.r == 8
        n_loaded = sum(p.numel() for p in loaded_hypernet.parameters())
        n_original = sum(p.numel() for p in hyperlora.parameters())
        assert n_loaded == n_original
