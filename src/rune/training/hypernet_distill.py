"""D2L privileged-context self-distillation for the HyperLoRA hypernetwork.

Teacher = frozen base model with the trajectory in-context (adapters disabled);
student = base + generated adapter with the trajectory removed from the prompt.
Loss = top-K KL over the answer span, masked to diff tokens (where teacher != base).

GPU imports are deferred; only pure tensor helpers are import-safe.
"""
from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100


def run_hypernet_distillation(config: Any) -> None:
    """Stage-2 entrypoint (D2L context distillation). Implemented in Task 11."""
    raise NotImplementedError("implemented in Task 11")


def compute_diff_positions(base_top1: Any, teacher_top1: Any, labels: Any) -> Any:
    """Boolean mask: supervised positions where base and teacher top-1 disagree."""
    return (labels != IGNORE_INDEX) & (base_top1 != teacher_top1)


def topk_kl_loss(student_logits: Any, teacher_logits: Any, k: int = 50) -> Any:
    """KL(teacher || student) over the teacher's top-K tokens, mean over rows.

    Args:
        student_logits: [N, V] student logits at supervised positions.
        teacher_logits: [N, V] teacher logits at the same positions.
        k: number of top teacher tokens to match.
    """
    import torch  # noqa: PLC0415

    k = min(k, teacher_logits.shape[-1])
    topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
    t_denom = torch.logsumexp(teacher_logits.float(), dim=-1, keepdim=True)
    teacher_logp = topk_vals.float() - t_denom  # [N, K]
    teacher_p = teacher_logp.exp()  # [N, K]
    s_denom = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
    student_logq = student_logits.float().gather(-1, topk_idx) - s_denom  # [N, K]
    return (teacher_p * (teacher_logp - student_logq)).sum(dim=-1).mean()


def distill_step_loss(
    student_logits: Any,
    teacher_logits: Any,
    base_top1: Any,
    teacher_top1: Any,
    labels: Any,
    k: int = 50,
) -> Any:
    """Top-K KL restricted to diff positions (base != teacher on supervised tokens).

    Returns a scalar loss. If there are no diff positions, returns 0 (no signal).
    """
    mask = compute_diff_positions(base_top1, teacher_top1, labels)
    if int(mask.sum()) == 0:
        return student_logits.sum() * 0.0
    return topk_kl_loss(student_logits[mask], teacher_logits[mask], k=k)
