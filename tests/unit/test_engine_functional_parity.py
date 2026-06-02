"""Toy logit-parity: engine PEFT apply path == functional contract apply path.

The reviewer's #1 concern: the engine's PEFT scaling
(``lora_alpha_peft = checkpoint_alpha * r_peft`` so PEFT's ``lora_alpha/r``
realizes the RAW checkpoint ``lora_alpha`` un-divided) is so far proven ONLY by
arithmetic (``test_engine_apply_scaling.py``), never by a real forward where a
generated adapter flows through ``peft.set_peft_model_state_dict`` and the PEFT
``LoraLayer`` forward.

This test closes that gap on CPU with tiny tensors by running BOTH real paths on
the SAME base weights + SAME input and asserting identical output logits:

  ENGINE path   — real ``peft.get_peft_model(LoraConfig(r=r_peft,
                  lora_alpha=lora_alpha_peft))`` sized by the real
                  ``wrapper.peft_scaling_params``; weights flattened by the real
                  ``hypernetwork._to_peft_state_dict``; loaded by real
                  ``peft.set_peft_model_state_dict``; logits from the real PEFT
                  forward.
  FUNCTIONAL path — real ``adapter_contract.assemble_adapter`` (head bias into
                  ranks ``r..2r-1`` when ``use_bias``) + ``lora_delta`` at real
                  ``effective_scaling`` (== checkpoint ``lora_alpha``).

Both the no_bias (``r_peft == r``) and use_bias (``r_peft == 2r``) cases are
covered. The use_bias case is the one that validates that PEFT applies the SAME
``alpha`` scaling uniformly across the bias half (ranks ``r..2r-1``) — the
"rank-16 crash" worry. fp32, tight tolerance; bf16/flash-attn numerics are the
GPU harness's job, not the toy's.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from rune.model.adapter_contract import assemble_adapter, effective_scaling, lora_delta
from rune.model.hypernetwork import _to_peft_state_dict
from rune.model.wrapper import peft_scaling_params

# Single target module (an MLP module name so _to_peft_state_dict routes it to
# `mlp.<name>` and _functional_lora-style routing agrees).
TARGET = "down_proj"


class _TinyAttn(torch.nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        # unused here, but present so the layer object has self_attn for parity
        self.q_proj = torch.nn.Linear(d, d, bias=False)


class _TinyMLP(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(d_in, d_out, bias=False)


class _TinyLayer(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.self_attn = _TinyAttn(d_in)
        self.mlp = _TinyMLP(d_in, d_out)


class _TinyInner(torch.nn.Module):
    """Mimics ``Qwen...Model``: holds ``.layers``."""

    def __init__(self, n_layers: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_TinyLayer(d_in, d_out) for _ in range(n_layers)]
        )


class _TinyCausalLM(torch.nn.Module):
    """Mimics ``Qwen...ForCausalLM``: ``.model.layers``.

    forward returns ``SimpleNamespace(logits=...)`` over the concatenated
    down_proj outputs of the selected layers so a wrong delta on ANY selected
    layer changes the logits (the parity target).
    """

    def __init__(self, n_layers: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.model = _TinyInner(n_layers, d_in, d_out)

    def forward(self, x: torch.Tensor, layer_indices: list[int]) -> Any:
        outs = [_down_proj(self, i)(x) for i in layer_indices]
        return SimpleNamespace(logits=torch.cat(outs, dim=-1))


def _down_proj(base: _TinyCausalLM, layer_idx: int) -> torch.nn.Linear:
    """Typed accessor for the layer's down_proj.

    ``ModuleList`` indexing returns a bare ``Module`` (and ``nn.Module``'s
    ``__getattr__`` widens attribute access to ``Tensor | Module``), so the
    ``.layers[i].mlp.down_proj`` chain loses its concrete type. Cast once here.
    """
    layer = cast(_TinyLayer, base.model.layers[layer_idx])
    return layer.mlp.down_proj


def _hyp(use_bias: bool, lora_alpha: float, bias: dict[str, Any] | None) -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(
            use_bias=use_bias,
            lora_config=SimpleNamespace(lora_alpha=lora_alpha),
        ),
        get_head_bias=lambda: bias,
    )


def _functional_logits(
    base: _TinyCausalLM,
    assembled: dict[str, dict[str, Any]],
    scaling: float,
    x: torch.Tensor,
    layer_indices: list[int],
) -> torch.Tensor:
    """Functional contract forward: base down_proj output + lora_delta per layer.

    Mirrors ``_functional_lora``'s positional slicing (``[:, layer_pos]``) and
    the shared ``lora_delta`` ((x @ Aᵀ) @ B * scaling), but on the toy module so
    it stays CPU-only and dtype-clean.
    """
    a = assembled[TARGET]["A"]
    b = assembled[TARGET]["B"]
    outs = []
    for layer_pos, layer_idx in enumerate(layer_indices):
        base_out = _down_proj(base, layer_idx)(x)
        delta = lora_delta(x, a[:, layer_pos], b[:, layer_pos], scaling)
        outs.append(base_out + delta.to(base_out.dtype))
    return torch.cat(outs, dim=-1)


def _run_parity(*, use_bias: bool) -> None:
    torch.manual_seed(0)
    checkpoint_alpha = 13.5
    rank = 4
    n_layers, d_in, d_out, seq = 3, 6, 5, 7
    layer_indices = list(range(n_layers))

    # Generated context A/B (the hypernet's `generate_weights` output shape).
    gen_a = torch.randn(1, n_layers, rank, d_in, dtype=torch.float32)
    gen_b = torch.randn(1, n_layers, rank, d_out, dtype=torch.float32)
    lora_dict = {TARGET: {"A": gen_a, "B": gen_b}}

    bias: dict[str, Any] | None = None
    if use_bias:
        # Head bias concatenated into ranks r..2r-1 by combine_lora; same shape
        # as a context A/B block.
        bias = {
            TARGET: {
                "A": torch.randn(n_layers, rank, d_in, dtype=torch.float32),
                "B": torch.randn(n_layers, rank, d_out, dtype=torch.float32),
            }
        }

    hyp = _hyp(use_bias=use_bias, lora_alpha=checkpoint_alpha, bias=bias)
    n_chunks = torch.ones(1, dtype=torch.int32)
    assembled = assemble_adapter(hyp, lora_dict, n_chunks)
    scaling = effective_scaling(hyp)
    assert scaling == checkpoint_alpha

    # --- shared base weights ---
    base = _TinyCausalLM(n_layers, d_in, d_out).to(torch.float32).eval()
    base_for_peft = copy.deepcopy(base)

    x = torch.randn(1, seq, d_in, dtype=torch.float32)

    # --- FUNCTIONAL path (shared contract) ---
    with torch.no_grad():
        func_logits = _functional_logits(base, assembled, scaling, x, layer_indices)

    # --- ENGINE path (real peft) ---
    r_peft, lora_alpha_peft = peft_scaling_params(checkpoint_alpha, rank, use_bias)
    expected_r = 2 * rank if use_bias else rank
    assert r_peft == expected_r
    # The whole contract: PEFT's per-layer scaling lora_alpha/r == checkpoint alpha.
    assert lora_alpha_peft / r_peft == pytest.approx(checkpoint_alpha)

    lora_config = LoraConfig(
        r=r_peft,
        lora_alpha=lora_alpha_peft,
        target_modules=[TARGET],
        lora_dropout=0.0,
        use_rslora=False,
    )
    # _TinyCausalLM is a CPU stand-in for a PreTrainedModel; get_peft_model only
    # needs the module structure (.model.layers + target_modules), not the HF base.
    peft_model = get_peft_model(base_for_peft, lora_config).eval()  # type: ignore[arg-type]

    # Flatten the SAME assembled adapter into PEFT keys exactly as the engine does.
    # _to_peft_state_dict expects A=[1,n_layers,r,d_in], B=[1,n_layers,r,d_out];
    # the assembled rank axis is r_peft (== 2r under use_bias). It stores
    # lora_A = A[0,pos] ([r_peft,d_in]) and lora_B = B[0,pos].t() ([d_out,r_peft]).
    flat = _to_peft_state_dict(assembled, layer_indices, [TARGET])
    load_result = set_peft_model_state_dict(peft_model, flat)
    # No unexpected/missing keys -> the engine flat dict fully populated the adapter.
    if load_result is not None:
        assert not getattr(load_result, "unexpected_keys", [])

    # The PEFT-wrapped model's forward signature: peft delegates to the base
    # forward, which here takes (x, layer_indices). get_peft_model adds the LoRA
    # delta inside each down_proj forward, so engine_logits already include it.
    with torch.no_grad():
        engine_logits = peft_model(x, layer_indices).logits

    max_abs = float((engine_logits - func_logits).abs().max())
    assert torch.allclose(engine_logits, func_logits, atol=1e-5, rtol=1e-4), (
        f"engine PEFT logits diverge from functional contract logits "
        f"(use_bias={use_bias}, max_abs_diff={max_abs:.3e})"
    )


def test_engine_peft_matches_functional_no_bias() -> None:
    # r_peft == r; scaling identity is trivial but the full PEFT forward path
    # (set_peft_model_state_dict + LoraLayer forward) must still equal the
    # functional contract delta.
    _run_parity(use_bias=False)


def test_engine_peft_matches_functional_use_bias() -> None:
    # r_peft == 2r: validates lora_alpha_peft = alpha*2r keeps PEFT's
    # alpha_peft/r_peft == alpha applied UNIFORMLY across the bias half
    # (ranks r..2r-1) — the reviewer's "is the bias half scaled the same?" worry.
    _run_parity(use_bias=True)
