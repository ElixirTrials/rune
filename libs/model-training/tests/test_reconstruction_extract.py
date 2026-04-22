"""Tests for PEFT state_dict → T2L-shape extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _make_lora_key(layer: int, prefix: str, module: str, ab: str) -> str:
    return f"base_model.model.model.layers.{layer}.{prefix}.{module}.lora_{ab}.weight"


def _fabricate_state_dict(
    *,
    layers: tuple[int, ...],
    module_prefix: dict[str, str],  # module -> "self_attn" or "mlp"
    rank: int,
    in_features: int,
    out_features: int,
    dtype: "torch.dtype" = None,  # type: ignore[assignment]
) -> dict[str, "torch.Tensor"]:
    if dtype is None:
        dtype = torch.float32
    sd: dict[str, torch.Tensor] = {}
    for mod, prefix in module_prefix.items():
        for layer in layers:
            # PEFT layout: A is (rank, in_features); B is (out_features, rank).
            sd[_make_lora_key(layer, prefix, mod, "A")] = torch.randn(
                rank, in_features, dtype=dtype
            )
            sd[_make_lora_key(layer, prefix, mod, "B")] = torch.randn(
                out_features, rank, dtype=dtype
            )
    return sd


def test_extract_shapes_and_layer_order() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(2, 0, 1),  # unsorted on purpose
        module_prefix={"q_proj": "self_attn", "gate_proj": "mlp"},
        rank=8,
        in_features=128,
        out_features=256,
    )
    out = extract_lora_ab_from_state_dict(
        sd, target_modules=("q_proj", "gate_proj")
    )
    assert set(out) == {"q_proj", "gate_proj"}
    for mod in ("q_proj", "gate_proj"):
        assert out[mod]["A"].shape == (3, 8, 128)
        assert out[mod]["B"].shape == (3, 256, 8)
        assert out[mod]["layer_indices"].tolist() == [0, 1, 2]


def test_extract_detects_layer_indices_per_module() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(0, 1, 2, 3),
        module_prefix={"q_proj": "self_attn"},
        rank=4,
        in_features=64,
        out_features=64,
    )
    out = extract_lora_ab_from_state_dict(sd, target_modules=("q_proj",))
    assert out["q_proj"]["layer_indices"].tolist() == [0, 1, 2, 3]


def test_extract_raises_when_module_has_no_keys() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(0, 1),
        module_prefix={"q_proj": "self_attn"},
        rank=4,
        in_features=32,
        out_features=32,
    )
    with pytest.raises(ValueError, match="no .* keys for module 'v_proj'"):
        extract_lora_ab_from_state_dict(sd, target_modules=("q_proj", "v_proj"))


def test_extract_raises_when_a_b_layer_sets_disagree() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(0, 1),
        module_prefix={"q_proj": "self_attn"},
        rank=4,
        in_features=32,
        out_features=32,
    )
    del sd[_make_lora_key(1, "self_attn", "q_proj", "B")]
    with pytest.raises(ValueError, match="lora_A .* lora_B .* mismatch"):
        extract_lora_ab_from_state_dict(sd, target_modules=("q_proj",))


def test_load_adapter_as_record(tmp_path: Path) -> None:
    from model_training.reconstruction.extract import load_adapter_as_record
    from safetensors.torch import save_file

    sd = _fabricate_state_dict(
        layers=(0, 1),
        module_prefix={"q_proj": "self_attn", "v_proj": "self_attn"},
        rank=4,
        in_features=32,
        out_features=32,
    )
    adapter_dir = tmp_path / "adapter-abc"
    adapter_dir.mkdir()
    save_file(sd, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "danielcherubini/Qwen3.5-DeltaCoder-9B",
                "r": 4,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM",
            }
        )
    )
    record_kwargs = load_adapter_as_record(
        adapter_dir,
        task_id="abc",
        task_description="some description",
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        base_model_id_override="Qwen/Qwen3.5-9B",
        created_at="2026-04-22T00:00:00Z",
    )
    assert record_kwargs["rank"] == 4
    assert record_kwargs["target_modules"] == ("q_proj", "v_proj")
    assert record_kwargs["layer_indices"] == (0, 1)
    assert record_kwargs["base_model_id"] == "Qwen/Qwen3.5-9B"
    assert (
        record_kwargs["warm_start_adapter"] == "danielcherubini/Qwen3.5-DeltaCoder-9B"
    )


def test_extract_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.extract")
    assert hasattr(mod, "extract_lora_ab_from_state_dict")
