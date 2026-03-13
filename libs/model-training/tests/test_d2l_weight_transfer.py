"""Tests for partial weight transfer functions in model_training.sakana_d2l.

CPU-only tests covering:
- transfer_aggregator_weights() freezes aggregator params (requires_grad=False)
- transfer_aggregator_weights() leaves head params trainable (requires_grad=True)
- transfer_aggregator_weights() result has missing_keys only for head.*
- _assert_transfer_integrity() raises AssertionError on missing aggregator key
- _assert_transfer_integrity() raises AssertionError on unexpected keys
- _assert_transfer_integrity() passes with clean transfer (head.* missing, no unexpected)
- get_aggregator_config() extracts aggregator_config from checkpoint dict
- get_aggregator_config() raises ValueError when aggregator_config is None
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fake model helpers
# ---------------------------------------------------------------------------


class _FakeHyperLoRA(nn.Module):
    """Minimal stand-in for HyperLoRA with aggregator.* and head.* params.

    Gives named parameters:
      aggregator.0.weight, aggregator.0.bias,
      aggregator.1.weight, aggregator.1.bias,
      head.weight, head.bias
    """

    def __init__(self) -> None:
        super().__init__()
        self.aggregator = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 16))
        self.head = nn.Linear(16, 12)


def _make_aggregator_checkpoint(model: _FakeHyperLoRA) -> dict[str, Any]:
    """Build a state-dict containing only aggregator.* tensors.

    Simulates the filtered Sakana checkpoint that contains aggregator weights
    but NOT head weights (head must be re-initialized for the new target model).
    """
    return {k: v for k, v in model.state_dict().items() if k.startswith("aggregator.")}


class _FakeIncompatibleKeys:
    """Simulates PyTorch's _IncompatibleKeys namedtuple from load_state_dict."""

    def __init__(self, missing_keys: list[str], unexpected_keys: list[str]) -> None:
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys


# ---------------------------------------------------------------------------
# Test 1: transfer_aggregator_weights freezes aggregator params
# ---------------------------------------------------------------------------


def test_transfer_freezes_aggregator_params(monkeypatch: Any) -> None:
    """After transfer_aggregator_weights(), all aggregator.* params are frozen."""
    from model_training.sakana_d2l import transfer_aggregator_weights

    target = _FakeHyperLoRA()
    source = _FakeHyperLoRA()
    fake_sd = _make_aggregator_checkpoint(source)

    monkeypatch.setattr(
        "model_training.sakana_d2l.torch.load",
        lambda *a, **kw: fake_sd,
    )

    transfer_aggregator_weights(target, "/fake/checkpoint.bin")

    for name, param in target.named_parameters():
        if name.startswith("aggregator."):
            assert param.requires_grad is False, (
                f"Expected aggregator param '{name}' to be frozen "
                f"(requires_grad=False), but got requires_grad=True"
            )


# ---------------------------------------------------------------------------
# Test 2: transfer_aggregator_weights leaves head params trainable
# ---------------------------------------------------------------------------


def test_transfer_leaves_head_trainable(monkeypatch: Any) -> None:
    """After transfer_aggregator_weights(), all head.* params remain trainable."""
    from model_training.sakana_d2l import transfer_aggregator_weights

    target = _FakeHyperLoRA()
    source = _FakeHyperLoRA()
    fake_sd = _make_aggregator_checkpoint(source)

    monkeypatch.setattr(
        "model_training.sakana_d2l.torch.load",
        lambda *a, **kw: fake_sd,
    )

    transfer_aggregator_weights(target, "/fake/checkpoint.bin")

    for name, param in target.named_parameters():
        if name.startswith("head."):
            assert param.requires_grad is True, (
                f"Expected head param '{name}' to remain trainable "
                f"(requires_grad=True), but got requires_grad=False"
            )


# ---------------------------------------------------------------------------
# Test 3: transfer_aggregator_weights result has head.* in missing_keys
# ---------------------------------------------------------------------------


def test_transfer_head_in_missing_keys(monkeypatch: Any) -> None:
    """After transfer, missing_keys contains only head.* keys (head not in checkpoint)."""
    from model_training.sakana_d2l import transfer_aggregator_weights

    target = _FakeHyperLoRA()
    source = _FakeHyperLoRA()
    fake_sd = _make_aggregator_checkpoint(source)

    # Capture what load_state_dict returns by intercepting _assert_transfer_integrity
    captured: dict[str, Any] = {}
    original_assert: Any = None

    def _capture_loaded(hypernet: Any, loaded: Any) -> None:
        captured["loaded"] = loaded
        if original_assert is not None:
            original_assert(hypernet, loaded)

    import model_training.sakana_d2l as sakana_mod

    original_assert = sakana_mod._assert_transfer_integrity
    monkeypatch.setattr(sakana_mod, "_assert_transfer_integrity", _capture_loaded)
    monkeypatch.setattr(
        "model_training.sakana_d2l.torch.load",
        lambda *a, **kw: fake_sd,
    )

    transfer_aggregator_weights(target, "/fake/checkpoint.bin")

    loaded = captured["loaded"]
    missing = loaded.missing_keys
    assert len(missing) > 0, "Expected some missing_keys (head.* not loaded)"
    for key in missing:
        assert key.startswith("head."), (
            f"Expected all missing_keys to be head.*, but found: '{key}'"
        )


# ---------------------------------------------------------------------------
# Test 4: _assert_transfer_integrity raises on missing aggregator key
# ---------------------------------------------------------------------------


def test_assert_integrity_fails_on_missing_aggregator() -> None:
    """_assert_transfer_integrity raises AssertionError if aggregator key missing."""
    from model_training.sakana_d2l import _assert_transfer_integrity

    model = _FakeHyperLoRA()
    # Simulate a transfer where an aggregator key was missing (bad checkpoint)
    loaded = _FakeIncompatibleKeys(
        missing_keys=["aggregator.0.weight", "head.weight", "head.bias"],
        unexpected_keys=[],
    )

    with pytest.raises(AssertionError):
        _assert_transfer_integrity(model, loaded)


# ---------------------------------------------------------------------------
# Test 5: _assert_transfer_integrity raises on unexpected keys
# ---------------------------------------------------------------------------


def test_assert_integrity_fails_on_unexpected_keys() -> None:
    """_assert_transfer_integrity raises AssertionError if unexpected_keys non-empty."""
    from model_training.sakana_d2l import _assert_transfer_integrity

    model = _FakeHyperLoRA()
    # All head.* missing is fine, but unexpected keys indicate a mismatch
    loaded = _FakeIncompatibleKeys(
        missing_keys=["head.weight", "head.bias"],
        unexpected_keys=["some_unknown_key"],
    )

    with pytest.raises(AssertionError):
        _assert_transfer_integrity(model, loaded)


# ---------------------------------------------------------------------------
# Test 6: _assert_transfer_integrity passes on clean transfer
# ---------------------------------------------------------------------------


def test_assert_integrity_passes_clean_transfer() -> None:
    """_assert_transfer_integrity does NOT raise when only head.* missing, no unexpected."""
    from model_training.sakana_d2l import _assert_transfer_integrity

    model = _FakeHyperLoRA()
    # Clean transfer: only head keys missing (not loaded from aggregator checkpoint)
    loaded = _FakeIncompatibleKeys(
        missing_keys=["head.weight", "head.bias"],
        unexpected_keys=[],
    )

    # Should not raise — this is the expected success case
    _assert_transfer_integrity(model, loaded)


# ---------------------------------------------------------------------------
# Test 7: get_aggregator_config returns config from checkpoint
# ---------------------------------------------------------------------------


def test_get_aggregator_config_returns_config(monkeypatch: Any) -> None:
    """get_aggregator_config extracts aggregator_config from fake checkpoint dict."""
    from model_training.sakana_d2l import get_aggregator_config

    fake_aggregator_config = {"num_layers": 4, "hidden_size": 256}

    fake_hypernet_config = MagicMock()
    fake_hypernet_config.aggregator_config = fake_aggregator_config

    fake_sd: dict[str, Any] = {"hypernet_config": fake_hypernet_config}

    monkeypatch.setattr(
        "model_training.sakana_d2l.torch.load",
        lambda *a, **kw: fake_sd,
    )

    result = get_aggregator_config("/fake/checkpoint.bin")
    assert result == fake_aggregator_config


# ---------------------------------------------------------------------------
# Test 8: get_aggregator_config raises ValueError on None aggregator_config
# ---------------------------------------------------------------------------


def test_get_aggregator_config_raises_on_none(monkeypatch: Any) -> None:
    """get_aggregator_config raises ValueError when checkpoint's aggregator_config is None."""
    from model_training.sakana_d2l import get_aggregator_config

    fake_hypernet_config = MagicMock()
    fake_hypernet_config.aggregator_config = None

    fake_sd: dict[str, Any] = {"hypernet_config": fake_hypernet_config}

    monkeypatch.setattr(
        "model_training.sakana_d2l.torch.load",
        lambda *a, **kw: fake_sd,
    )

    with pytest.raises(ValueError, match="aggregator_config is None"):
        get_aggregator_config("/fake/checkpoint.bin")
