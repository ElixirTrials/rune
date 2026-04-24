"""Across-corpus z-score statistics for reconstruction targets.

Mirrors T2L's ``std_recon_target + 1e-10`` normalization (see
``hyper_llm_modulator/recon_trainer.py``). Stats are computed element-wise
across the task dimension so a downstream hypernetwork with ``pred_z_score``
can normalize oracle targets before the L1 loss.

Inputs are dicts of
``{module: {"A": Tensor[n_layers, r, in], "B": Tensor[n_layers, out, r]}}``
as produced by ``extract.extract_lora_ab_from_state_dict``. All records must
share the same module set and per-module tensor shapes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

STD_FLOOR = 1e-6


def compute_zscore_stats(
    per_record_tensors: Iterable[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Compute element-wise ``{avg_A, std_A, avg_B, std_B}`` per module.

    Args:
        per_record_tensors: Iterable of per-record dicts
            ``{module: {"A": Tensor, "B": Tensor}}``. At least one record
            required; all records must share the same module keys and
            per-module tensor shapes.

    Returns:
        ``{module: {"avg_A": Tensor, "std_A": Tensor, "avg_B": Tensor,
        "std_B": Tensor}}``. ``std`` tensors are floored to ``STD_FLOOR``
        for numerical safety.

    Raises:
        ValueError: On empty iterable or inconsistent module sets across records.
    """
    import torch  # noqa: PLC0415

    records = list(per_record_tensors)
    if not records:
        raise ValueError("compute_zscore_stats requires at least one record")

    reference_modules = set(records[0].keys())
    for i, rec in enumerate(records[1:], start=1):
        if set(rec.keys()) != reference_modules:
            raise ValueError(
                f"inconsistent modules at record {i}: "
                f"expected {sorted(reference_modules)}, got {sorted(rec.keys())}"
            )

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for module in reference_modules:
        a_stack = torch.stack([rec[module]["A"] for rec in records], dim=0).float()
        b_stack = torch.stack([rec[module]["B"] for rec in records], dim=0).float()
        # n_records > 1 → use population std (unbiased=False) so the stat is
        # finite even with 2 samples. Downstream z-score normalization is
        # what matters; tiny-N variance is bounded by STD_FLOOR anyway.
        avg_a = a_stack.mean(dim=0)
        avg_b = b_stack.mean(dim=0)
        std_a = a_stack.std(dim=0, unbiased=False).clamp_min(STD_FLOOR)
        std_b = b_stack.std(dim=0, unbiased=False).clamp_min(STD_FLOOR)
        stats[module] = {
            "avg_A": avg_a,
            "std_A": std_a,
            "avg_B": avg_b,
            "std_B": std_b,
        }
    return stats


def save_zscore_stats(stats: dict[str, dict[str, Any]], path: Path) -> None:
    """Persist the stats dict via ``torch.save``.

    Args:
        stats: Stats dict as returned by ``compute_zscore_stats``.
        path: Destination file path (parent dirs created if absent).
    """
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, str(path))


def load_zscore_stats(path: Path) -> dict[str, dict[str, Any]]:
    """Inverse of ``save_zscore_stats``.

    Args:
        path: Path to a ``.pt`` file written by ``save_zscore_stats``.

    Returns:
        Stats dict ``{module: {"avg_A", "std_A", "avg_B", "std_B"}}``.

    Raises:
        ValueError: If the loaded object is not a dict.
    """
    import torch  # noqa: PLC0415

    loaded = torch.load(str(path), weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected dict in {path}, got {type(loaded)!r}")
    return loaded


__all__ = [
    "STD_FLOOR",
    "compute_zscore_stats",
    "save_zscore_stats",
    "load_zscore_stats",
]
