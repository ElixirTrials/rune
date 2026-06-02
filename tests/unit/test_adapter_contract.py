"""Parity tests for the shared adapter-apply contract (rune.model.adapter_contract).

Verifies that the single source of truth for adapter assembly + effective scaling
matches Sakana's ctx_to_lora convention on toy tensors:

(a) assemble_adapter with use_bias appends head-bias as extra rank slices to A and B
    (combined rank == 2r) and leaves ranks 0..r-1 unchanged.
(b) effective_scaling == lora_config.lora_alpha (NOT alpha/r).
(c) the functional delta via the shared path EQUALS lora_layer.lora_forward's delta
    on the same toy A/B/x/scaling (numerical-equivalence, atol 1e-5).
(d) grad flows through assemble_adapter to a requires_grad head-bias.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from rune.model.adapter_contract import (
    assemble_adapter,
    effective_scaling,
    lora_delta,
)


def _fake_hyp(use_bias: bool, lora_alpha: float, bias: dict[str, Any] | None) -> Any:
    cfg = SimpleNamespace(
        use_bias=use_bias,
        lora_config=SimpleNamespace(lora_alpha=lora_alpha),
    )
    return SimpleNamespace(config=cfg, get_head_bias=lambda: bias)


def test_assemble_adapter_use_bias_appends_rank_slices() -> None:
    n_layers, r, d_in, d_out = 3, 4, 6, 5
    gen_a = torch.randn(1, n_layers, r, d_in)
    gen_b = torch.randn(1, n_layers, r, d_out)
    lora_dict = {"down_proj": {"A": gen_a, "B": gen_b}}

    bias_a = torch.randn(n_layers, r, d_in)
    bias_b = torch.randn(n_layers, r, d_out)
    bias = {"down_proj": {"A": bias_a, "B": bias_b}}

    hyp = _fake_hyp(use_bias=True, lora_alpha=10.0, bias=bias)
    n_chunks = torch.ones(1, dtype=torch.int32)

    out = assemble_adapter(hyp, lora_dict, n_chunks)
    out_a = out["down_proj"]["A"]
    out_b = out["down_proj"]["B"]

    # combined rank doubles to 2r
    assert out_a.shape == (1, n_layers, 2 * r, d_in)
    assert out_b.shape == (1, n_layers, 2 * r, d_out)
    # ranks 0..r-1 unchanged (context A/B)
    torch.testing.assert_close(out_a[0, :, :r, :], gen_a[0])
    torch.testing.assert_close(out_b[0, :, :r, :], gen_b[0])
    # ranks r..2r-1 are the head bias
    torch.testing.assert_close(out_a[0, :, r : 2 * r, :], bias_a)
    torch.testing.assert_close(out_b[0, :, r : 2 * r, :], bias_b)


def test_assemble_adapter_no_bias_is_identity() -> None:
    n_layers, r, d_in, d_out = 2, 3, 5, 4
    gen_a = torch.randn(1, n_layers, r, d_in)
    gen_b = torch.randn(1, n_layers, r, d_out)
    lora_dict = {"down_proj": {"A": gen_a, "B": gen_b}}

    hyp = _fake_hyp(use_bias=False, lora_alpha=10.0, bias=None)
    n_chunks = torch.ones(1, dtype=torch.int32)

    out = assemble_adapter(hyp, lora_dict, n_chunks)
    assert out["down_proj"]["A"].shape == (1, n_layers, r, d_in)
    torch.testing.assert_close(out["down_proj"]["A"], gen_a)
    torch.testing.assert_close(out["down_proj"]["B"], gen_b)


def test_effective_scaling_is_lora_alpha_not_alpha_over_r() -> None:
    hyp = _fake_hyp(use_bias=True, lora_alpha=45.2548, bias=None)
    # lora_config.r is intentionally absent — effective_scaling must NOT divide by r.
    assert effective_scaling(hyp) == 45.2548
    assert isinstance(effective_scaling(hyp), float)


def test_lora_delta_matches_sakana_lora_forward() -> None:
    from ctx_to_lora.modeling.lora_layer import lora_forward  # noqa: PLC0415

    torch.manual_seed(0)
    r, d_in, d_out, seq = 4, 6, 5, 7
    scaling = 3.5
    x = torch.randn(1, seq, d_in)
    a = torch.randn(1, r, d_in)
    b = torch.randn(1, r, d_out)

    ours = lora_delta(x, a, b, scaling)

    linear = torch.nn.Linear(d_in, d_out)
    linear.eval()
    n_qs = torch.tensor([1])
    full = lora_forward(
        x,
        n_qs,
        1,
        a,
        b,
        0.0,  # lora_dropout_p
        scaling,
        linear,
    )
    ref_delta = full - torch.nn.Linear.forward(linear, x)

    torch.testing.assert_close(ours, ref_delta, atol=1e-5, rtol=1e-5)


def test_assemble_adapter_grad_flows_to_head_bias() -> None:
    n_layers, r, d_in, d_out = 2, 3, 5, 4
    gen_a = torch.randn(1, n_layers, r, d_in)
    gen_b = torch.randn(1, n_layers, r, d_out)
    lora_dict = {"down_proj": {"A": gen_a, "B": gen_b}}

    bias_a = torch.randn(n_layers, r, d_in, requires_grad=True)
    bias_b = torch.randn(n_layers, r, d_out, requires_grad=True)
    bias = {"down_proj": {"A": bias_a, "B": bias_b}}

    hyp = _fake_hyp(use_bias=True, lora_alpha=10.0, bias=bias)
    n_chunks = torch.ones(1, dtype=torch.int32)

    out = assemble_adapter(hyp, lora_dict, n_chunks)
    out["down_proj"]["A"].sum().backward()
    assert bias_a.grad is not None
    assert bias_a.grad.abs().sum() > 0
