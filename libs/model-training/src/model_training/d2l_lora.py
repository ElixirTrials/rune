"""Functional LoRA injection via context manager for hypernetwork training.

Patches transformer attention projection modules with F.linear forward
functions that carry live hypernetwork tensor graph nodes, preserving
autograd continuity through A and B matrices back to the hypernetwork head.

Unlike PEFT's get_peft_model (which severs the autograd graph by copying
tensors into new nn.Parameter objects), this approach uses closures over
the original A/B tensors so that loss.backward() propagates gradients
through the LoRA path all the way to the hypernetwork parameters.

All heavy GPU imports (torch, transformers) are deferred to function bodies
per INFRA-05 project convention.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = ["apply_functional_lora"]


def _extract_layer_idx(module_path: str, layer_indices: list[int]) -> int | None:
    """Extract positional index within layer_indices from a dotted module path.

    Finds the last numeric segment in the path (e.g. "model.layers.7.self_attn.q_proj"
    yields 7), then returns its position in layer_indices. Returns None if the
    absolute layer index is not found in layer_indices.

    Args:
        module_path: Dotted module name from model.named_modules().
        layer_indices: Ordered list of absolute layer indices to patch.

    Returns:
        Positional index into layer_indices, or None if not a target layer.
    """
    abs_idx = next(
        (int(p) for p in reversed(module_path.split(".")) if p.isdigit()),
        None,
    )
    if abs_idx is None:
        return None
    try:
        return list(layer_indices).index(abs_idx)
    except ValueError:
        return None


def _make_lora_forward(
    weight: Any, bias: Any, lora_a: Any, lora_b: Any, scale: float
) -> Callable[..., Any]:
    """Create a patched forward function that adds LoRA to the base linear.

    The returned closure computes:
        base_out = F.linear(x, w_frozen, bias_frozen)
        lora_out = F.linear(F.linear(x, lora_a), lora_b.t()) * scale
        return base_out + lora_out

    weight and bias are detached so that base model parameters do not receive
    gradients. lora_a and lora_b remain live graph nodes.

    Args:
        weight: Original weight tensor (will be detached).
        bias: Original bias tensor or None (will be detached if present).
        lora_a: LoRA down-projection tensor, shape (r, d_in). Must have grad enabled.
        lora_b: LoRA up-projection tensor, shape (r, d_out). Must have grad enabled.
        scale: LoRA scaling factor (lora_alpha / r).

    Returns:
        Callable that replaces module.forward.
    """
    import torch.nn.functional as func  # noqa: PLC0415

    w_frozen = weight.detach()
    bias_frozen = bias.detach() if bias is not None else None

    def patched_forward(x: Any) -> Any:
        base_out = func.linear(x, w_frozen, bias_frozen)
        lora_ax = func.linear(x, lora_a)
        lora_out = func.linear(lora_ax, lora_b.t()) * scale
        return base_out + lora_out

    return patched_forward


class _FunctionalLoRAContext:
    """Context manager that patches model modules with functional LoRA forwards.

    On enter: iterates model.named_modules(), finds modules whose short name
    is in target_modules and whose layer index is in hc.layer_indices, saves
    their original forward methods, and replaces them with LoRA-augmented
    closures.

    On exit: restores all original forward methods via try/finally.
    """

    def __init__(self, model: Any, lora_dict: dict[str, Any], hc: Any) -> None:
        self._model = model
        self._lora_dict = lora_dict
        self._hc = hc
        self._patched_modules: list[tuple[str, Any]] = []

    def __enter__(self) -> _FunctionalLoRAContext:
        r: int = self._hc.lora_config.r
        scale: float = self._hc.lora_config.lora_alpha / r
        target_modules: list[str] = list(self._hc.lora_config.target_modules)
        layer_indices_list: list[int] = list(self._hc.layer_indices)

        for module_path, module in self._model.named_modules():
            short_name = module_path.split(".")[-1]
            if short_name not in target_modules:
                continue

            layer_pos = _extract_layer_idx(module_path, layer_indices_list)
            if layer_pos is None:
                continue

            # Get A and B for this module type and layer position
            lora_a = self._lora_dict[short_name]["A"][0, layer_pos]  # (r, d_in)
            lora_b = self._lora_dict[short_name]["B"][0, layer_pos]  # (r, d_out)

            weight = module.weight

            # Shape validation
            if lora_a.shape[1] != weight.shape[1]:
                msg = (
                    f"Shape mismatch at '{module_path}': "
                    f"A.shape[1]={lora_a.shape[1]} != W.shape[1]={weight.shape[1]}"
                )
                raise RuntimeError(msg)
            if lora_b.shape[1] != weight.shape[0]:
                msg = (
                    f"Shape mismatch at '{module_path}': "
                    f"B.shape[1]={lora_b.shape[1]} != W.shape[0]={weight.shape[0]}"
                )
                raise RuntimeError(msg)

            bias = getattr(module, "bias", None)

            # Track module for restoration (we'll delete the instance override)
            self._patched_modules.append((module_path, module))
            module.forward = _make_lora_forward(weight, bias, lora_a, lora_b, scale)  # type: ignore[method-assign]

            logger.debug("Patched %s (layer_pos=%d)", module_path, layer_pos)

        logger.info("Functional LoRA applied to %d modules", len(self._patched_modules))
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            for _path, module in self._patched_modules:
                # Remove the instance-level forward override, restoring
                # the class-level nn.Linear.forward descriptor
                module.__dict__.pop("forward", None)
        finally:
            self._patched_modules.clear()
        logger.debug("All forwards restored")
        return None


def apply_functional_lora(
    model: Any, lora_dict: dict[str, Any], hc: Any
) -> _FunctionalLoRAContext:
    """Create a context manager that patches model with functional LoRA.

    Usage:
        with apply_functional_lora(base_model, lora_dict, hc):
            output = base_model(input_ids)
            loss = criterion(output, target)
        loss.backward()  # gradients flow through A/B to hypernetwork

    Args:
        model: Base transformer model (nn.Module).
        lora_dict: Dict from HyperLoRA.generate_weights(). Structure:
            lora_dict[proj_name]["A"] shape: (batch=1, n_layers, r, d_in)
            lora_dict[proj_name]["B"] shape: (batch=1, n_layers, r, d_out)
            Keys match hc.lora_config.target_modules.
            Batch dimension is always 1 (squeezed at index 0).
        hc: HypernetConfig with attributes:
            hc.lora_config.target_modules: list of projection names
            hc.lora_config.r: LoRA rank
            hc.lora_config.lora_alpha: scaling numerator
            hc.layer_indices: list of absolute layer indices

    Returns:
        _FunctionalLoRAContext to use as context manager.
    """
    return _FunctionalLoRAContext(model, lora_dict, hc)
