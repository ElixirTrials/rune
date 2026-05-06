"""Tests for per-layer Frobenius norm computation."""
from __future__ import annotations

import torch
import pytest

from scripts.paper.frobenius_norm import compute_frobenius_norms


def test_nonzero_norms() -> None:
    """Non-zero adapter weights produce non-zero norms."""
    state_dict = {
        "layer.0.lora_A": torch.randn(8, 64),
        "layer.0.lora_B": torch.randn(64, 8),
        "layer.1.lora_A": torch.randn(8, 64),
        "layer.1.lora_B": torch.randn(64, 8),
    }
    norms = compute_frobenius_norms(state_dict)
    assert len(norms) == 4
    assert all(v > 0.0 for v in norms.values())


def test_zero_adapter_zero_norm() -> None:
    """Zero weights produce zero norm."""
    state_dict = {"layer.0.lora_A": torch.zeros(8, 64)}
    norms = compute_frobenius_norms(state_dict)
    assert norms["layer.0.lora_A"] == 0.0
