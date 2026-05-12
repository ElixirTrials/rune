"""Tests for KD loss top1_agreement metric in train_hypernet_hpo.py."""

import sys
from importlib import import_module
from pathlib import Path

import torch

# scripts/ is not on sys.path by default in pytest; add it so the import works
# without triggering the bootstrap.setup_path() that requires GPU libs.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# torch.utils.checkpoint raises RuntimeError if CUDA is first initialized inside
# a checkpoint forward pass (e.g. via autocast("cuda")).  Touch CUDA once at
# module load time so it is already initialised before any test runs.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")


def _load_loss_fns():
    mod = import_module("train_hypernet_hpo")
    return mod._chunked_kl_ce_loss, mod._full_kl_ce_loss


def test_top1_agreement_perfect_match():
    """When student == teacher, top-1 agreement should be 1.0."""
    chunked, full = _load_loss_fns()
    logits = torch.randn(1, 10, 100, requires_grad=True)
    teacher = logits.detach().clone()

    _, metrics_c = chunked(logits, teacher, alpha=0.5, temperature=1.0)
    assert "top1_agreement" in metrics_c
    assert metrics_c["top1_agreement"] == 1.0

    _, metrics_f = full(logits, teacher, alpha=0.5, temperature=1.0)
    assert "top1_agreement" in metrics_f
    assert metrics_f["top1_agreement"] == 1.0


def test_top1_agreement_different_logits():
    """When student and teacher disagree on every token, agreement should be 0."""
    chunked, full = _load_loss_fns()
    teacher = torch.zeros(1, 10, 100)
    teacher[0, :, 0] = 10.0

    student = torch.zeros(1, 10, 100, requires_grad=True)
    with torch.no_grad():
        student[0, :, 1] = 10.0

    _, metrics_c = chunked(student, teacher, alpha=0.5, temperature=1.0)
    assert metrics_c["top1_agreement"] == 0.0

    _, metrics_f = full(student, teacher, alpha=0.5, temperature=1.0)
    assert metrics_f["top1_agreement"] == 0.0


def test_top1_agreement_partial():
    """Half matching tokens should give 0.5 agreement."""
    chunked, full = _load_loss_fns()
    teacher = torch.zeros(1, 10, 100)
    teacher[0, :, 0] = 10.0

    student = torch.zeros(1, 10, 100, requires_grad=True)
    with torch.no_grad():
        student[0, :5, 0] = 10.0
        student[0, 5:, 1] = 10.0

    _, metrics_c = chunked(student, teacher, alpha=0.5, temperature=1.0)
    assert metrics_c["top1_agreement"] == 0.5

    _, metrics_f = full(student, teacher, alpha=0.5, temperature=1.0)
    assert metrics_f["top1_agreement"] == 0.5


def test_metrics_dict_keys():
    """Both loss functions must return all expected metric keys."""
    chunked, full = _load_loss_fns()
    logits = torch.randn(1, 4, 50, requires_grad=True)
    teacher = torch.randn(1, 4, 50)

    expected_keys = {"kl_loss", "ce_loss", "total_loss", "top1_agreement"}

    _, m_c = chunked(logits, teacher, alpha=0.9, temperature=1.0)
    assert set(m_c.keys()) == expected_keys

    _, m_f = full(logits, teacher, alpha=0.9, temperature=1.0)
    assert set(m_f.keys()) == expected_keys
