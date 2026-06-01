"""CPU unit tests for the engine PEFT export/apply scaling contract.

The engine hot-swaps generated LoRA weights into a PEFT-wrapped base model.
PEFT applies the delta as ``delta * (lora_alpha / r)``. To reproduce the
shared contract's effective scaling (the RAW checkpoint ``lora_alpha``, applied
un-divided — see rune.model.adapter_contract.effective_scaling), the engine's
PEFT LoraConfig must satisfy ``lora_alpha_peft / r_peft == checkpoint_alpha``.

These tests pin the pure arithmetic and rank sizing that wrapper.from_config
builds, plus the merge_head_bias_rank guard that combine_lora's rank-doubling
relies on. They do NOT exercise a real model — functional-vs-PEFT logit parity
on a real checkpoint must be run as a GPU anchor next.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rune.model.hypernetwork import _peft_adapter_rank, merge_head_bias_rank
from rune.model.wrapper import peft_scaling_params as _engine_peft_params


@pytest.mark.parametrize(
    ("checkpoint_alpha", "rank"),
    [(16.0, 8), (45.2548, 16), (32.0, 32)],
)
def test_peft_scaling_reproduces_contract_no_bias(
    checkpoint_alpha: float, rank: int
) -> None:
    r_peft, lora_alpha_peft = _engine_peft_params(checkpoint_alpha, rank, False)
    assert r_peft == rank
    # PEFT scaling alpha_peft / r_peft must equal the contract effective scaling.
    assert lora_alpha_peft / r_peft == pytest.approx(checkpoint_alpha)


@pytest.mark.parametrize(
    ("checkpoint_alpha", "rank"),
    [(16.0, 8), (45.2548, 16), (32.0, 32)],
)
def test_peft_scaling_reproduces_contract_use_bias(
    checkpoint_alpha: float, rank: int
) -> None:
    r_peft, lora_alpha_peft = _engine_peft_params(checkpoint_alpha, rank, True)
    # use_bias doubles the rank axis (combine_lora packs head bias into r..2r-1).
    assert r_peft == 2 * rank
    # Scaling is uniform across all 2r ranks and equals the contract scaling.
    assert lora_alpha_peft / r_peft == pytest.approx(checkpoint_alpha)


def test_use_bias_does_not_change_effective_scaling() -> None:
    # The whole point: turning on bias changes r_peft AND alpha_peft together so
    # the realized scaling is identical to the no-bias case (== checkpoint alpha).
    _, a_nobias = _engine_peft_params(16.0, 8, False)
    r_nobias, _ = _engine_peft_params(16.0, 8, False)
    r_bias, a_bias = _engine_peft_params(16.0, 8, True)
    assert a_nobias / r_nobias == pytest.approx(a_bias / r_bias)


def test_merge_head_bias_rank_accepts_doubled_rank() -> None:
    # use_bias checkpoint: combine_lora makes rank base_rank+bias_rank == 2r,
    # PEFT is built at 2r -> guard must pass (the rank-16 crash is gone).
    assert (
        merge_head_bias_rank(adapter_rank=16, bias_rank=16, peft_config_rank=32) == 32
    )


def test_merge_head_bias_rank_rejects_checkpoint_r() -> None:
    # Passing the checkpoint r (16) instead of the PEFT r (32) is the old bug;
    # the guard must still catch genuine rank drift.
    with pytest.raises(ValueError, match="recreate the PEFT adapter"):
        merge_head_bias_rank(adapter_rank=16, bias_rank=16, peft_config_rank=16)


def test_peft_adapter_rank_reads_active_adapter() -> None:
    model = SimpleNamespace(
        peft_config={"default": SimpleNamespace(r=32)},
        active_adapter="default",
    )
    assert _peft_adapter_rank(model, fallback=999) == 32


def test_peft_adapter_rank_falls_back_without_config() -> None:
    model = SimpleNamespace(peft_config=None, active_adapter=None)
    assert _peft_adapter_rank(model, fallback=32) == 32


def test_peft_adapter_rank_falls_back_when_active_missing() -> None:
    # active_adapter not a key -> use the first config value rather than crash.
    model = SimpleNamespace(
        peft_config={"default": SimpleNamespace(r=24)},
        active_adapter=None,
    )
    assert _peft_adapter_rank(model, fallback=999) == 24
