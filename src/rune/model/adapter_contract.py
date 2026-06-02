"""Shared adapter-apply contract (single source of truth).

Rune's recall collapse was an APPLICATION bug: bespoke paths (1) scaled the LoRA
delta by ``alpha/r`` (8x too weak) and (2) skipped ``combine_lora`` / head-bias
assembly. Sakana's convention (``ctx_to_lora.modeling.lora_layer.lora_forward`` +
``hypernet.patch_lora_forward``) is:

    delta = (x @ Aᵀ) @ B * scaling   with   scaling == lora_config.lora_alpha

applied to the ``combine_lora``-assembled adapter (ranks 0..r-1 = context A/B,
ranks r..2r-1 = head bias when ``use_bias`` is set).

This module is the ONE place every apply site routes through so the contract can
never drift per-caller again. The delta math itself is NOT re-implemented here —
it is imported from ``rune.training.hypernet_distill._lora_delta`` so there is a
single einsum implementation in the repo.
"""

from __future__ import annotations

from typing import Any


def assemble_adapter(hyp: Any, lora_dict: dict[str, Any], n_chunks: Any) -> Any:
    """Assemble the effective adapter, appending head bias as extra rank slices.

    Mirrors Sakana's inference path: when the hypernet was trained ``use_bias``,
    ``combine_lora`` concatenates the head bias into ranks ``r..2r-1`` (context A/B
    stay in ranks ``0..r-1``), doubling the rank axis. With ``use_bias`` off the
    raw ``lora_dict`` is returned unchanged (``combine_lora`` with no bias would
    only zero-pad). Autograd-safe: ``combine_lora`` assigns generated/bias tensors
    into a fresh ``torch.zeros`` via differentiable slice writes, so gradients flow
    back to both the context weights and ``get_head_bias`` parameters.

    Args:
        hyp: Loaded HyperLoRA model exposing ``config.use_bias`` and
            ``get_head_bias()``.
        lora_dict: ``{module: {"A", "B"}}`` with A/B shaped
            ``[n_ctx, n_layers, r, dim]`` from ``generate_weights``.
        n_chunks: ``[n_ctx]`` int tensor of chunks per context (typically
            ``torch.ones(1)`` for a single context).

    Returns:
        The assembled ``{module: {"A", "B"}}`` (rank ``2r`` if ``use_bias`` else
        ``r``).
    """
    if not getattr(hyp.config, "use_bias", False):
        return lora_dict
    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415

    return combine_lora(lora_dict, n_chunks, lora_bias=hyp.get_head_bias())


def effective_scaling(hyp: Any) -> float:
    """Return the Sakana-parity effective scaling: ``lora_config.lora_alpha``.

    NOT ``alpha/r``. This is the same attribute Sakana applies un-divided in
    ``lora_forward`` (``scaling = peft_config.lora_alpha``). Engine PEFT configs
    must realize this same effective value (``lora_alpha_peft / r_peft == alpha``).
    """
    return float(hyp.config.lora_config.lora_alpha)


def lora_delta(x: Any, a: Any, b: Any, scaling: float) -> Any:
    """LoRA delta ``(x @ Aᵀ) @ B * scaling`` for a single context (n_ctx=1).

    Re-export of the canonical einsum implementation so every apply site uses one
    math path. a: ``[1, r, d_in]``, b: ``[1, r, d_out]``, x: ``[1, seq, d_in]`` ->
    ``[1, seq, d_out]``.
    """
    from rune.training.hypernet_distill import _lora_delta  # noqa: PLC0415

    return _lora_delta(x, a, b, scaling)
