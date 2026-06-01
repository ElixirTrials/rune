import pytest
import torch

from rune.model.hypernetwork import _to_peft_state_dict, merge_head_bias_rank


def test_peft_keys_match_expected_pattern_and_no_truncation() -> None:
    r, d = 4, 8
    # HyperLoRA emits both A and B as [bs, n_layers, r, dim] (rank at dim -2);
    # combine_lora rearranges B identically to A, so B arrives as [r, out].
    lora_dict = {"q_proj": {"A": torch.randn(1, 2, r, d), "B": torch.randn(1, 2, r, d)}}
    sd = _to_peft_state_dict(lora_dict, layer_indices=[0, 1], target_modules=["q_proj"])
    a_keys = [k for k in sd if k.endswith("lora_A.weight")]
    b_keys = [k for k in sd if k.endswith("lora_B.weight")]
    assert len(a_keys) == 2 and len(b_keys) == 2
    # B must be transposed to [out, r]; A stays [r, in]
    assert sd[a_keys[0]].shape == (r, d)
    assert sd[b_keys[0]].shape == (d, r)


def test_peft_export_maps_positional_slot_to_absolute_layer_noncontiguous() -> None:
    # Closes the reviewer gap: the GPU parity smoke only exercised contiguous
    # layers 0-31. Both the PEFT export (_to_peft_state_dict) and the training
    # functional path (_functional_lora) iterate enumerate(layer_indices), pulling
    # tensor slot [layer_pos] for absolute layer {layer_idx}. Verify that mapping
    # holds for NON-CONTIGUOUS layers so positional!=absolute can't silently
    # misapply weights at inference.
    r, d = 2, 3
    layer_indices = [0, 5, 10]  # positional slots 0,1,2 -> absolute layers 0,5,10
    a = torch.arange(1 * 3 * r * d, dtype=torch.float32).reshape(1, 3, r, d)
    b = torch.arange(1 * 3 * r * d, dtype=torch.float32).reshape(1, 3, r, d) + 100.0
    sd = _to_peft_state_dict({"q_proj": {"A": a, "B": b}}, layer_indices, ["q_proj"])

    for pos, layer in enumerate(layer_indices):
        ka = f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"
        kb = f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight"
        # absolute layer key pulls the POSITIONAL slot, not the slot==layer index
        assert torch.equal(sd[ka], a[0, pos])
        assert torch.equal(sd[kb], b[0, pos].t())
    # slot/layer divergence is real: layer-5 key must NOT equal slot 5 (doesn't exist)
    assert torch.equal(sd[
        "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight"
    ], a[0, 1])


def test_merge_head_bias_rank_raises_on_rank_mismatch() -> None:
    # Combining a rank-b bias into a rank-r adapter changes effective rank;
    # the PEFT config rank must match or we must raise (no silent misapply).
    with pytest.raises(ValueError, match="rank"):
        merge_head_bias_rank(adapter_rank=4, bias_rank=2, peft_config_rank=4)


def test_merge_head_bias_rank_ok_when_config_matches_combined() -> None:
    assert merge_head_bias_rank(adapter_rank=4, bias_rank=2, peft_config_rank=6) == 6
