"""Tests for adapter placement mask."""

import torch

from model_training.adapter_generator import apply_placement_mask


def test_apply_placement_mask_noop():
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(2),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.ones(2),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.ones(2),
    }
    result = apply_placement_mask(state_dict, target_modules=None, layer_selection="all")
    assert result is state_dict


def test_apply_placement_mask_early_half():
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.ones(2),
    }
    result = apply_placement_mask(
        state_dict, target_modules=None, layer_selection="early_half",
        total_layers=2,
    )
    assert torch.all(result["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] == 1.0)
    assert torch.all(result["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"] == 0.0)


def test_apply_placement_mask_module_filter():
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2),
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.ones(2),
    }
    result = apply_placement_mask(
        state_dict, target_modules=["q_proj"], layer_selection="all",
    )
    assert torch.all(result["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] == 1.0)
    assert torch.all(result["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"] == 0.0)


def test_apply_placement_mask_late_half():
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.ones(2),
    }
    result = apply_placement_mask(
        state_dict, target_modules=None, layer_selection="late_half",
        total_layers=2,
    )
    assert torch.all(result["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] == 0.0)
    assert torch.all(result["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"] == 1.0)
