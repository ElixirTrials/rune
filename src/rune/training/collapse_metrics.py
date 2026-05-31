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


def diff_token_fraction(base_top1: Any, teacher_top1: Any) -> float:
    """Fraction of supervised positions where teacher disagrees with base.

    The online analogue of the teacher-quality audit's diff-token fraction — logged
    over real training rows to confirm the sampled audit was representative
    (reviewer). Returns 0.0 for an empty span.
    """
    n = int(teacher_top1.numel())
    if n == 0:
        return 0.0
    return int((base_top1 != teacher_top1).sum()) / n


def preservation_agreement(
    student_top1: Any, teacher_top1: Any, base_top1: Any
) -> float:
    """Student↔teacher agreement on the NON-diff (preservation) region.

    Diff-token training narrows the objective to base≠teacher positions; a broad
    perturbation can lift diff_agreement while damaging the much larger region
    where base already agrees with teacher (reviewer). This measures that region:
    fraction of positions where base==teacher AND student also ==teacher. A healthy
    adapter keeps this near 1.0. Returns 1.0 when there is no preservation region
    (vacuously preserved).
    """
    mask = base_top1 == teacher_top1
    denom = int(mask.sum())
    if denom == 0:
        return 1.0
    kept = int(((student_top1 == teacher_top1) & mask).sum())
    return kept / denom


def should_early_stop(
    step: int,
    warmup: int,
    recent_diff_agreement: list[float],
    recent_preservation: list[float],
    skipped: int,
    total: int,
    *,
    min_diff_agreement: float,
    min_preservation: float,
    max_skip_frac: float,
) -> str | None:
    """Return an abort reason after warmup, else None (reviewer: pre-defined stop).

    Guards against a long run whose loss falls for the wrong reason:
      - too many records skipped (no gradient reaching the hypernet),
      - diff_agreement still ~0 after warmup (adapter learning nothing useful),
      - preservation collapsed (adapter is a broad perturbation damaging the
        agreement region).
    """
    if step < warmup:
        return None
    if total > 0 and skipped / total > max_skip_frac:
        return f"skip_frac {skipped / total:.2f} > {max_skip_frac}"
    if recent_diff_agreement:
        mean_da = sum(recent_diff_agreement) / len(recent_diff_agreement)
        if mean_da < min_diff_agreement:
            return f"diff_agreement {mean_da:.3f} < {min_diff_agreement} after warmup"
    if recent_preservation:
        mean_pres = sum(recent_preservation) / len(recent_preservation)
        if mean_pres < min_preservation:
            return (
                f"preservation {mean_pres:.3f} < {min_preservation} "
                "(broad perturbation)"
            )
    return None
