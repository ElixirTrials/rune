"""Tests for adapter_generator module (CPU-only, no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor(shape: tuple[int, ...]) -> MagicMock:
    """Return a MagicMock that looks like a tensor with the given shape."""
    t = MagicMock()
    t.shape = shape
    t.contiguous.return_value = t
    t.t.return_value = t
    return t


def _make_hc(
    layer_indices: list[int] | None = None,
    target_modules: list[str] | None = None,
    rank: int = 8,
    use_bias: bool = False,
) -> MagicMock:
    """Return a MagicMock HypernetConfig."""
    hc = MagicMock()
    hc.layer_indices = layer_indices or [0, 1]
    hc.lora_config.target_modules = target_modules or ["q_proj", "v_proj"]
    hc.lora_config.r = rank
    hc.lora_config.lora_alpha = rank * 2
    hc.use_bias = use_bias
    return hc


def _make_lora_dict(
    target_modules: list[str], num_layers: int = 2, rank: int = 8
) -> dict[str, dict[str, Any]]:
    """Return a minimal lora_dict matching what combine_lora might return."""
    d: dict[str, dict[str, Any]] = {}
    for mod in target_modules:
        d[mod] = {
            "A": _make_tensor((1, num_layers, rank, 64)),
            "B": _make_tensor((1, num_layers, rank, 128)),
        }
    return d


# ---------------------------------------------------------------------------
# _save_adapter
# ---------------------------------------------------------------------------


def test_save_adapter_produces_peft_files(tmp_path: Path) -> None:
    """_save_adapter writes adapter_model.safetensors and adapter_config.json."""
    from model_training.adapter_generator import _save_adapter

    target_modules = ["q_proj", "v_proj"]
    layer_indices = [0, 2]
    hc = _make_hc(layer_indices=layer_indices, target_modules=target_modules, rank=8)
    lora_dict = _make_lora_dict(target_modules, num_layers=len(layer_indices), rank=8)

    saved_files: dict[str, Any] = {}

    def fake_save_file(tensors: dict[str, Any], path: str) -> None:
        saved_files["tensors"] = tensors
        saved_files["path"] = path

    with patch("safetensors.torch.save_file", side_effect=fake_save_file):
        _save_adapter(
            lora_dict=lora_dict,
            output_dir=str(tmp_path),
            base_model_name="google/gemma-2b",
            hc=hc,
            scaling_factor=0.16,
        )

    # adapter_model.safetensors was requested
    assert saved_files["path"] == str(tmp_path / "adapter_model.safetensors")

    # Key format: base_model.model.model.layers.{i}.{prefix}.{mod}.lora_{X}.weight
    for mod in target_modules:
        for layer_idx in layer_indices:
            prefix = "self_attn"
            assert (
                f"base_model.model.model.layers.{layer_idx}.{prefix}.{mod}.lora_A.weight"
                in saved_files["tensors"]
            )
            assert (
                f"base_model.model.model.layers.{layer_idx}.{prefix}.{mod}.lora_B.weight"
                in saved_files["tensors"]
            )

    # adapter_config.json is written
    config_path = tmp_path / "adapter_config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["peft_type"] == "LORA"
    assert config["r"] == 8
    assert config["task_type"] == "CAUSAL_LM"
    assert config["base_model_name_or_path"] == "google/gemma-2b"
    assert set(config["target_modules"]) == set(target_modules)
    assert "lora_alpha" in config


# ---------------------------------------------------------------------------
# generate_adapter — pool mode
# ---------------------------------------------------------------------------


def test_generate_adapter_pool_mode(tmp_path: Path) -> None:
    """Pool mode: borrows from pool, saves adapter, returns output_dir."""
    from model_training.adapter_generator import generate_adapter

    target_modules = ["q_proj", "v_proj"]
    layer_indices = [0, 1]
    rank = 8
    hc = _make_hc(
        layer_indices=layer_indices, target_modules=target_modules, rank=rank
    )

    # Build fake hypernet
    fake_hypernet = MagicMock()
    fake_hypernet.config.use_bias = False
    fake_hypernet.get_head_bias.return_value = None
    lora_dict = _make_lora_dict(
        target_modules, num_layers=len(layer_indices), rank=rank
    )
    layernorm_dict: dict[str, Any] = {}
    fake_hypernet.generate_weights.return_value = (lora_dict, layernorm_dict)

    # Build fake pool
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_pool = MagicMock()
    fake_pool.hypernetwork.return_value = (fake_hypernet, hc)
    fake_pool.base_model.return_value = (fake_model, fake_tokenizer)
    fake_pool.model_name = "google/gemma-2b"

    # Fake features/attn_mask
    fake_features = MagicMock()
    fake_attn_mask = MagicMock()

    saved_paths: list[str] = []

    def fake_save_file(tensors: dict[str, Any], path: str) -> None:
        saved_paths.append(path)

    with (
        patch(
            "model_training.d2l_probe.extract_activations_with_model",
            return_value=(fake_features, fake_attn_mask),
        ),
        patch(
            "ctx_to_lora.modeling.lora_merger.combine_lora",
            return_value=lora_dict,
        ),
        patch("safetensors.torch.save_file", side_effect=fake_save_file),
        patch(
            "torch.no_grad",
            return_value=MagicMock(
                __enter__=lambda s: None,
                __exit__=lambda s, *a: None,
            ),
        ),
        patch("torch.ones", return_value=MagicMock()),
    ):
        result = generate_adapter(
            text="some trajectory text",
            output_dir=str(tmp_path),
            pool=fake_pool,
        )

    assert result == str(tmp_path)
    fake_pool.hypernetwork.assert_called_once()
    fake_pool.base_model.assert_called_once()

    # Verify adapter_config.json was written
    config_path = tmp_path / "adapter_config.json"
    assert config_path.exists()


# ---------------------------------------------------------------------------
# generate_adapter — standalone mode
# ---------------------------------------------------------------------------


def test_generate_adapter_standalone_mode(tmp_path: Path) -> None:
    """Standalone mode: loads hypernet, extracts activations, saves adapter."""
    from model_training.adapter_generator import generate_adapter

    target_modules = ["q_proj", "v_proj"]
    layer_indices = [0, 1]
    rank = 8
    hc = _make_hc(
        layer_indices=layer_indices, target_modules=target_modules, rank=rank
    )

    fake_hypernet = MagicMock()
    fake_hypernet.config.use_bias = False
    fake_hypernet.get_head_bias.return_value = None
    lora_dict = _make_lora_dict(
        target_modules, num_layers=len(layer_indices), rank=rank
    )
    layernorm_dict: dict[str, Any] = {}
    fake_hypernet.generate_weights.return_value = (lora_dict, layernorm_dict)

    fake_features = MagicMock()
    fake_attn_mask = MagicMock()

    load_hypernet_calls: list[dict[str, Any]] = []

    def fake_load_hypernetwork(
        checkpoint_path: Any = None,
        variant: str = "gemma_demo",
        device: str = "cpu",
    ) -> tuple[Any, Any]:
        load_hypernet_calls.append(
            {"checkpoint_path": checkpoint_path, "variant": variant, "device": device}
        )
        return fake_hypernet, hc

    extract_calls: list[dict[str, Any]] = []

    def fake_extract_activations(
        text: str,
        base_model_name: str,
        layer_indices: list[int],
        device: str = "cpu",
        max_length: int = 512,
    ) -> tuple[Any, Any]:
        extract_calls.append(
            {
                "text": text,
                "base_model_name": base_model_name,
                "layer_indices": layer_indices,
            }
        )
        return fake_features, fake_attn_mask

    saved_paths: list[str] = []

    def fake_save_file(tensors: dict[str, Any], path: str) -> None:
        saved_paths.append(path)

    import model_training.adapter_generator as ag_mod

    with (
        patch.object(
            ag_mod,
            "extract_activations",
            side_effect=fake_extract_activations,
        ),
        patch(
            "model_training.adapter_generator._generate_adapter_standalone",
            wraps=ag_mod._generate_adapter_standalone,
        ),
        patch("safetensors.torch.save_file", side_effect=fake_save_file),
        patch(
            "torch.no_grad",
            return_value=MagicMock(
                __enter__=lambda s: None,
                __exit__=lambda s, *a: None,
            ),
        ),
        patch("torch.ones", return_value=MagicMock()),
        patch(
            "ctx_to_lora.modeling.lora_merger.combine_lora",
            return_value=lora_dict,
        ),
    ):
        import model_training.hypernetwork as _hn_mod
        with patch.object(
            _hn_mod,
            "load_hypernetwork",
            side_effect=fake_load_hypernetwork,
        ):
            result = generate_adapter(
                text="test trajectory",
                output_dir=str(tmp_path),
                pool=None,
                checkpoint_path="/tmp/hn.pt",
                base_model_name="google/gemma-2b",
                variant="gemma_demo",
                device="cpu",
            )

    assert result == str(tmp_path)
    assert len(load_hypernet_calls) == 1
    assert load_hypernet_calls[0]["checkpoint_path"] == "/tmp/hn.pt"
    assert len(extract_calls) == 1
    assert extract_calls[0]["base_model_name"] == "google/gemma-2b"

    config_path = tmp_path / "adapter_config.json"
    assert config_path.exists()


# ---------------------------------------------------------------------------
# extract_activations — empty text
# ---------------------------------------------------------------------------


def test_extract_activations_empty_text_raises() -> None:
    """extract_activations raises ValueError for empty or whitespace-only text."""
    from model_training.adapter_generator import extract_activations

    with pytest.raises(ValueError, match="empty text"):
        extract_activations(
            text="",
            base_model_name="google/gemma-2b",
            layer_indices=[0, 1],
        )

    with pytest.raises(ValueError, match="empty text"):
        extract_activations(
            text="   ",
            base_model_name="google/gemma-2b",
            layer_indices=[0, 1],
        )
