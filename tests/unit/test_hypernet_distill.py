import torch

from rune.training.hypernet_distill import (
    compute_diff_positions,
    topk_kl_loss,
)


def test_compute_diff_positions_masks_to_labeled_disagreements() -> None:
    base_top1 = torch.tensor([5, 5, 5, 5])
    teacher_top1 = torch.tensor([5, 9, 7, 5])  # differ at 1,2
    labels = torch.tensor([1, 1, -100, 1])     # pos 2 unsupervised
    mask = compute_diff_positions(base_top1, teacher_top1, labels)
    assert mask.tolist() == [False, True, False, False]


def test_topk_kl_loss_zero_when_student_equals_teacher() -> None:
    teacher_logits = torch.randn(3, 10)
    # student identical -> KL ~ 0
    loss = topk_kl_loss(teacher_logits.clone(), teacher_logits, k=5)
    assert float(loss) < 1e-5


def test_topk_kl_loss_positive_when_distributions_differ() -> None:
    teacher_logits = torch.zeros(3, 10)
    teacher_logits[:, 0] = 10.0  # teacher confident on token 0
    student_logits = torch.zeros(3, 10)
    student_logits[:, 1] = 10.0  # student confident on token 1
    loss = topk_kl_loss(student_logits, teacher_logits, k=5)
    assert float(loss) > 0.5
