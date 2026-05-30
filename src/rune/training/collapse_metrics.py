"""Pure metric helpers for detecting hypernetwork adapter collapse.

No GPU imports at module load (CPU-importable invariant). torch is imported
lazily inside function bodies.
"""
from __future__ import annotations

from typing import Any


def assert_optimizer_covers(named_params: dict[str, Any], optimizer: Any) -> None:
    """Raise RuntimeError listing trainable params absent from the optimizer.

    Args:
        named_params: mapping of name -> nn.Parameter that must be optimized.
        optimizer: a torch optimizer whose param_groups are checked.
    """
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [name for name, p in named_params.items() if id(p) not in covered]
    if missing:
        msg = f"optimizer does not cover trainable params: {sorted(missing)}"
        raise RuntimeError(msg)


def summarize_named_tensors(named_tensors: dict[str, Any]) -> dict[str, float]:
    """Per-name mean/absmax/l2 stats for watched tensor groups."""
    out: dict[str, float] = {}
    for name, t in named_tensors.items():
        tf = t.detach().float()
        out[f"{name}/mean"] = float(tf.mean())
        out[f"{name}/absmax"] = float(tf.abs().max())
        out[f"{name}/l2"] = float(tf.norm())
    return out


def diff_agreement(student_top1: Any, teacher_top1: Any, base_top1: Any) -> float:
    """Fraction of diff positions where student matches teacher.

    Diff positions = where base_top1 != teacher_top1 (the tokens the trajectory
    is responsible for). Returns 0.0 when there are no diff positions.
    """
    mask = base_top1 != teacher_top1
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    agree = int(((student_top1 == teacher_top1) & mask).sum())
    return agree / denom
