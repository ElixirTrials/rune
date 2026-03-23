"""Adapter merging strategies for evolutionary combination.

Implements TIES-Merging and DARE-Merging for combining multiple LoRA
adapter state dicts into a single merged adapter. All GPU imports are
deferred inside function bodies (INFRA-05 pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ties_merge(
    state_dicts: list[dict[str, Any]],
    density: float = 0.5,
) -> dict[str, Any]:
    """Merge adapter state dicts using TIES-Merging.

    Trim-Elect-Sign-Disjoint merge: for each parameter, trims values
    below density threshold, elects the majority sign (ignoring trimmed
    values), then averages only the values matching the elected sign.

    Args:
        state_dicts: List of state dicts (tensors) to merge.
        density: Fraction of values to keep per parameter (0.0 to 1.0).

    Returns:
        Merged state dict with same keys and shapes as inputs.

    Raises:
        ValueError: If *state_dicts* is empty or *density* is outside
            ``[0.0, 1.0]``.
    """
    if not state_dicts:
        raise ValueError(
            "state_dicts must not be empty; TIES-merge of zero inputs is undefined."
        )
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be between 0.0 and 1.0, got {density}")

    import torch

    merged: dict[str, Any] = {}
    keys = state_dicts[0].keys()

    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors)

        # Track which values survived trimming (True = kept)
        keep_mask = torch.ones_like(stacked, dtype=torch.bool)

        # Trim: zero out values below density threshold per tensor
        for i in range(len(tensors)):
            flat = stacked[i].abs().flatten()
            if flat.numel() == 0:
                continue
            k_keep = max(1, int(flat.numel() * density))
            threshold = torch.topk(flat, k_keep).values[-1]
            mask = stacked[i].abs() >= threshold
            keep_mask[i] = mask
            stacked[i] = stacked[i] * mask.float()

        # Elect sign: majority vote using only surviving (non-trimmed) values
        signs = torch.sign(stacked)
        signs_masked = signs * keep_mask.float()
        sign_sum = signs_masked.sum(dim=0)
        elected_sign = torch.sign(sign_sum)

        # Where elected_sign == 0 (true tie), average all surviving values
        tie_mask = elected_sign == 0

        # Disjoint merge: average only values matching elected sign
        matching = (signs == elected_sign.unsqueeze(0)).float()
        # For tied positions, include all values that survived trimming
        matching = torch.where(
            tie_mask.unsqueeze(0),
            keep_mask.float(),
            matching,
        )
        matching_vals = stacked * matching
        count = matching.sum(dim=0).clamp(min=1)
        merged[key] = (matching_vals.sum(dim=0) / count).to(tensors[0].dtype)

    return merged


def dare_merge(
    state_dicts: list[dict[str, Any]],
    drop_rate: float = 0.1,
) -> dict[str, Any]:
    """Merge adapter state dicts using DARE-Merging.

    Drop-And-REscale merge: randomly drops a fraction of values from
    each state dict, then averages the remaining values with rescaling.

    Args:
        state_dicts: List of state dicts to merge.
        drop_rate: Fraction of values to drop per parameter. Must be in
            ``[0.0, 1.0)``.

    Returns:
        Merged state dict with same keys and shapes as inputs.

    Raises:
        ValueError: If *state_dicts* is empty or *drop_rate* is outside
            ``[0.0, 1.0)``.
    """
    if not state_dicts:
        raise ValueError(
            "state_dicts must not be empty; DARE-merge of zero inputs is undefined."
        )
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError(f"drop_rate must be in [0.0, 1.0), got {drop_rate}")

    import torch

    merged: dict[str, Any] = {}
    keys = state_dicts[0].keys()
    scale = 1.0 / (1.0 - drop_rate)

    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors)

        # Drop and rescale each tensor
        for i in range(len(tensors)):
            if drop_rate > 0:
                mask = (torch.rand_like(stacked[i]) >= drop_rate).float()
                stacked[i] = stacked[i] * mask * scale

        # Average across state dicts
        merged[key] = stacked.mean(dim=0).to(tensors[0].dtype)

    return merged


def load_adapter_state_dict(adapter_path: str | Path) -> dict[str, Any]:
    """Load a LoRA adapter state dict from a safetensors file.

    Args:
        adapter_path: Path to the adapter .safetensors file.

    Returns:
        State dict mapping parameter names to tensors.
    """
    from safetensors.torch import load_file

    return load_file(str(adapter_path), device="cpu")
