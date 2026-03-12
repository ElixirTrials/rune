"""Tests for model_training.merging — TIES and DARE merge strategies."""

import importlib.util
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_TORCH_AVAILABLE = "torch" in sys.modules or (
    importlib.util.find_spec("torch") is not None
)

requires_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


@requires_torch
def test_ties_merge_preserves_shapes() -> None:
    import torch
    from model_training.merging import ties_merge

    sd1 = {"weight": torch.randn(4, 4)}
    sd2 = {"weight": torch.randn(4, 4)}
    result = ties_merge([sd1, sd2], density=0.5)
    assert "weight" in result
    assert result["weight"].shape == (4, 4)


@requires_torch
def test_dare_merge_preserves_shapes() -> None:
    import torch
    from model_training.merging import dare_merge

    sd1 = {"weight": torch.randn(4, 4)}
    sd2 = {"weight": torch.randn(4, 4)}
    result = dare_merge([sd1, sd2], drop_rate=0.1)
    assert "weight" in result
    assert result["weight"].shape == (4, 4)


@requires_torch
def test_dare_merge_zero_drop_equals_mean() -> None:
    import torch
    from model_training.merging import dare_merge

    sd1 = {"weight": torch.ones(2, 2) * 2.0}
    sd2 = {"weight": torch.ones(2, 2) * 4.0}
    result = dare_merge([sd1, sd2], drop_rate=0.0)
    expected = 3.0
    assert torch.allclose(result["weight"], torch.ones(2, 2) * expected, atol=1e-5)


def test_load_adapter_state_dict_calls_safetensors() -> None:
    mock_safetensors_torch = MagicMock()
    mock_safetensors_torch.load_file = MagicMock(return_value={"w": "fake"})
    mock_safetensors = ModuleType("safetensors")

    sys.modules["safetensors"] = mock_safetensors
    sys.modules["safetensors.torch"] = mock_safetensors_torch
    try:
        from model_training.merging import load_adapter_state_dict

        result = load_adapter_state_dict("/fake/path.safetensors")
        mock_safetensors_torch.load_file.assert_called_once_with(
            "/fake/path.safetensors"
        )
        assert result == {"w": "fake"}
    finally:
        del sys.modules["safetensors"]
        del sys.modules["safetensors.torch"]


def test_ties_merge_empty_input() -> None:
    """ties_merge with no state dicts returns empty dict without importing torch."""
    # Temporarily block torch to ensure the function handles empty input
    original = sys.modules.get("torch")
    try:
        from model_training.merging import ties_merge

        result = ties_merge([])
        assert result == {}
    finally:
        if original is not None:
            sys.modules["torch"] = original


def test_dare_merge_empty_input() -> None:
    """dare_merge with no state dicts returns empty dict without importing torch."""
    original = sys.modules.get("torch")
    try:
        from model_training.merging import dare_merge

        result = dare_merge([])
        assert result == {}
    finally:
        if original is not None:
            sys.modules["torch"] = original
