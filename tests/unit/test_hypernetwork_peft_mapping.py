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


def test_merge_head_bias_rank_raises_on_rank_mismatch() -> None:
    # Combining a rank-b bias into a rank-r adapter changes effective rank;
    # the PEFT config rank must match or we must raise (no silent misapply).
    with pytest.raises(ValueError, match="rank"):
        merge_head_bias_rank(adapter_rank=4, bias_rank=2, peft_config_rank=4)


def test_merge_head_bias_rank_ok_when_config_matches_combined() -> None:
    assert merge_head_bias_rank(adapter_rank=4, bias_rank=2, peft_config_rank=6) == 6
