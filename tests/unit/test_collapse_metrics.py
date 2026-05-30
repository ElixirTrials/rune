import pytest
import torch

from rune.training.collapse_metrics import (
    assert_optimizer_covers,
    diff_agreement,
    summarize_named_tensors,
)


def test_assert_optimizer_covers_flags_missing_scaler_b() -> None:
    p_in = torch.nn.Parameter(torch.zeros(2))
    p_missing = torch.nn.Parameter(torch.zeros(2))
    named = {"scaler_B": p_missing, "head": p_in}
    opt = torch.optim.SGD([p_in], lr=0.1)
    try:
        assert_optimizer_covers(named, opt)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "scaler_B" in str(exc)


def test_assert_optimizer_covers_passes_when_all_present() -> None:
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p1, p2], lr=0.1)
    assert_optimizer_covers({"a": p1, "b": p2}, opt)  # no raise


def test_diff_agreement_zero_when_student_equals_base() -> None:
    base = torch.tensor([1, 1, 1, 1])
    teacher = torch.tensor([1, 2, 3, 1])  # differs at positions 1,2
    student = base.clone()  # student == base everywhere
    # top1_agreement(student, teacher) is high (2/4), but diff_agreement must be 0.
    assert diff_agreement(student, teacher, base) == 0.0


def test_diff_agreement_one_when_student_matches_teacher_on_diffs() -> None:
    base = torch.tensor([1, 1, 1, 1])
    teacher = torch.tensor([1, 2, 3, 1])
    student = teacher.clone()
    assert diff_agreement(student, teacher, base) == 1.0


def test_summarize_named_tensors_reports_absmax() -> None:
    stats = summarize_named_tensors({"scaler_B": torch.tensor([0.0, -0.013, 0.005])})
    assert stats["scaler_B/absmax"] == pytest.approx(0.013)
    assert "scaler_B/mean" in stats
    assert "scaler_B/l2" in stats
