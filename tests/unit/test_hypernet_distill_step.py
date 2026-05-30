import torch

from rune.training.hypernet_distill import distill_step_loss


def test_distill_step_loss_nonzero_and_backprops_to_scaler() -> None:
    torch.manual_seed(0)
    n, v = 4, 16
    teacher_logits = torch.zeros(n, v)
    teacher_logits[:, 3] = 8.0                 # teacher wants token 3
    base_top1 = torch.zeros(n, dtype=torch.long)  # base wants token 0 -> all diff
    teacher_top1 = torch.full((n,), 3, dtype=torch.long)
    labels = torch.ones(n, dtype=torch.long)

    scaler_b = torch.nn.Parameter(torch.ones(1))
    student_logits = torch.zeros(n, v) + scaler_b * 0.0  # depends on scaler_b
    student_logits = student_logits.clone()
    student_logits[:, 3] = scaler_b * 1.0       # student logit on token 3 scales with gate

    loss = distill_step_loss(student_logits, teacher_logits, base_top1, teacher_top1, labels)
    assert float(loss) > 0.0
    loss.backward()
    assert scaler_b.grad is not None and float(scaler_b.grad.abs().sum()) > 0.0
