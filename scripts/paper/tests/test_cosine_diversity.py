"""Tests for inter-adapter cosine diversity metric (Eq. 3)."""
from __future__ import annotations

import torch
import pytest

from scripts.paper.cosine_diversity import compute_cosine_diversity


def test_identical_adapters_zero_diversity() -> None:
    """Identical adapters should have diversity close to 0."""
    adapter = torch.randn(64, 128)
    adapters = [adapter, adapter.clone(), adapter.clone()]
    diversity = compute_cosine_diversity(adapters)
    assert diversity < 0.01


def test_orthogonal_adapters_high_diversity() -> None:
    """Orthogonal adapters should have diversity close to 1."""
    a = torch.zeros(2, 4)
    a[0, 0] = 1.0
    b = torch.zeros(2, 4)
    b[0, 1] = 1.0
    c = torch.zeros(2, 4)
    c[0, 2] = 1.0
    diversity = compute_cosine_diversity([a, b, c])
    assert diversity > 0.9


def test_single_adapter_returns_zero() -> None:
    """Single adapter should return diversity 0."""
    adapter = torch.randn(32, 64)
    assert compute_cosine_diversity([adapter]) == 0.0
