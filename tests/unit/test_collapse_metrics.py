import pytest
import torch

from rune.training.collapse_metrics import (
    assert_optimizer_covers,
    diff_agreement,
    diff_token_fraction,
    preservation_agreement,
    should_early_stop,
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


def test_diff_token_fraction() -> None:
    base = torch.tensor([1, 1, 1, 1])
    teacher = torch.tensor([1, 2, 3, 1])  # differs at 2 of 4
    assert diff_token_fraction(base, teacher) == 0.5


def test_preservation_agreement_high_when_student_keeps_agreement_region() -> None:
    base = torch.tensor([5, 5, 5, 9])
    teacher = torch.tensor([5, 5, 5, 1])  # agreement region = first 3 positions
    student = torch.tensor([5, 5, 5, 0])  # keeps all 3 agreement positions
    assert preservation_agreement(student, teacher, base) == 1.0


def test_preservation_agreement_drops_when_adapter_breaks_agreement_region() -> None:
    base = torch.tensor([5, 5, 5, 5])
    teacher = torch.tensor([5, 5, 5, 5])  # all agreement
    student = torch.tensor([5, 9, 9, 9])  # breaks 3 of 4
    assert preservation_agreement(student, teacher, base) == 0.25


def test_should_early_stop_none_during_warmup() -> None:
    assert (
        should_early_stop(
            10,
            100,
            [0.0],
            [0.0],
            100,
            100,
            min_diff_agreement=0.02,
            min_preservation=0.5,
            max_skip_frac=0.5,
        )
        is None
    )


def test_should_early_stop_fires_on_collapsed_preservation() -> None:
    reason = should_early_stop(
        200,
        100,
        [0.3],
        [0.1],
        0,
        100,
        min_diff_agreement=0.02,
        min_preservation=0.5,
        max_skip_frac=0.5,
    )
    assert reason is not None and "preservation" in reason


def test_should_early_stop_fires_on_high_skip_frac() -> None:
    reason = should_early_stop(
        200,
        100,
        [0.3],
        [0.9],
        80,
        100,
        min_diff_agreement=0.02,
        min_preservation=0.5,
        max_skip_frac=0.5,
    )
    assert reason is not None and "skip_frac" in reason


def test_should_early_stop_passes_when_healthy() -> None:
    assert (
        should_early_stop(
            200,
            100,
            [0.3],
            [0.9],
            5,
            100,
            min_diff_agreement=0.02,
            min_preservation=0.5,
            max_skip_frac=0.5,
        )
        is None
    )


def test_summarize_named_tensors_reports_absmax() -> None:
    stats = summarize_named_tensors({"scaler_B": torch.tensor([0.0, -0.013, 0.005])})
    assert stats["scaler_B/absmax"] == pytest.approx(0.013)
    assert "scaler_B/mean" in stats
    assert "scaler_B/l2" in stats
