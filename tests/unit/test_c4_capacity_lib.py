"""Rank-stacking composition math for the C4 capacity curve."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "_c4_capacity_lib.py"
_spec = importlib.util.spec_from_file_location("_c4_capacity_lib", _TOOL)
assert _spec is not None and _spec.loader is not None
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)


def _sd(a: torch.Tensor, b: torch.Tensor) -> dict:
    return {"m.lora_A.weight": a, "m.lora_B.weight": b}


def _delta(sd: dict) -> torch.Tensor:
    return sd["m.lora_B.weight"] @ sd["m.lora_A.weight"]


def test_rank_stacked_equals_sum_of_context_products_plus_one_bias() -> None:
    torch.manual_seed(0)
    ctx, bias, din, dout = 2, 1, 6, 5
    a_bias, b_bias = torch.randn(bias, din), torch.randn(dout, bias)
    a1, b1 = torch.randn(ctx, din), torch.randn(dout, ctx)
    a2, b2 = torch.randn(ctx, din), torch.randn(dout, ctx)
    sd1 = _sd(torch.cat([a1, a_bias]), torch.cat([b1, b_bias], dim=1))
    sd2 = _sd(torch.cat([a2, a_bias]), torch.cat([b2, b_bias], dim=1))
    comp = lib.compose_rank_stacked([sd1, sd2], ctx_rank=ctx)
    want = b1 @ a1 + b2 @ a2 + b_bias @ a_bias
    assert comp["m.lora_A.weight"].shape == (2 * ctx + bias, din)
    assert torch.allclose(_delta(comp), want, atol=1e-6)


def test_compose_single_adapter_is_identity() -> None:
    torch.manual_seed(1)
    sd = _sd(torch.randn(4, 6), torch.randn(5, 4))
    comp = lib.compose_rank_stacked([sd], ctx_rank=2)
    assert torch.equal(_delta(comp), _delta(sd))


def test_pad_adapter_rank_is_numerically_inert() -> None:
    torch.manual_seed(2)
    sd = _sd(torch.randn(3, 6), torch.randn(5, 3))
    padded = lib.pad_adapter_rank(sd, target_rank=10)
    assert padded["m.lora_A.weight"].shape == (10, 6)
    assert padded["m.lora_B.weight"].shape == (5, 10)
    assert torch.equal(_delta(padded), _delta(sd))


def test_pad_rejects_shrinking() -> None:
    sd = _sd(torch.zeros(4, 6), torch.zeros(5, 4))
    with pytest.raises(ValueError):
        lib.pad_adapter_rank(sd, target_rank=3)


def test_make_bundles_consecutive_disjoint_drops_remainder() -> None:
    assert lib.make_bundles(60, 1) == [[i] for i in range(60)]
    b8 = lib.make_bundles(60, 8)
    assert len(b8) == 7 and b8[0] == list(range(8)) and b8[-1] == list(range(48, 56))
    flat = [i for b in b8 for i in b]
    assert len(set(flat)) == len(flat) == 56


def test_campaign_rank() -> None:
    assert lib.campaign_rank(8, 8, 8) == 72
    assert lib.campaign_rank(8, 0, 8) == 64


def test_multi_cond_text_joins_blocks() -> None:
    assert lib.multi_cond_text(["a", "b"]) == "a\n\nb"
