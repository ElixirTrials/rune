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
            "/fake/path.safetensors", device="cpu"
        )
        assert result == {"w": "fake"}
    finally:
        del sys.modules["safetensors"]
        del sys.modules["safetensors.torch"]


def test_ties_merge_empty_input() -> None:
    """ties_merge raises ValueError for empty input (undefined operation)."""
    from model_training.merging import ties_merge

    with pytest.raises(ValueError, match="state_dicts must not be empty"):
        ties_merge([])


def test_dare_merge_empty_input() -> None:
    """dare_merge raises ValueError for empty input (undefined operation)."""
    from model_training.merging import dare_merge

    with pytest.raises(ValueError, match="state_dicts must not be empty"):
        dare_merge([])


@requires_torch
def test_ties_merge_density_validation() -> None:
    """ties_merge raises ValueError for density outside [0.0, 1.0]."""
    import torch
    from model_training.merging import ties_merge

    sd = [{"weight": torch.randn(4, 4)}]
    with pytest.raises(ValueError, match="density must be between"):
        ties_merge(sd, density=1.5)
    with pytest.raises(ValueError, match="density must be between"):
        ties_merge(sd, density=-0.1)


@requires_torch
def test_ties_merge_density_boundary_values() -> None:
    """density=0.0 and density=1.0 should not raise."""
    import torch
    from model_training.merging import ties_merge

    sd = [{"weight": torch.randn(4, 4)}, {"weight": torch.randn(4, 4)}]
    ties_merge(sd, density=0.0)
    ties_merge(sd, density=1.0)


@requires_torch
def test_ties_merge_sign_tie_averages_survivors() -> None:
    """When sign votes tie, TIES should average survivors, not drop all."""
    import torch
    from model_training.merging import ties_merge

    # One positive, one negative of equal magnitude -> tie
    sd1 = {"weight": torch.tensor([[1.0, 2.0]])}
    sd2 = {"weight": torch.tensor([[-1.0, -2.0]])}
    result = ties_merge([sd1, sd2], density=1.0)
    assert result["weight"].shape == (1, 2)
    # With tie fallback, averages both survivors -> 0.0
    assert torch.allclose(result["weight"], torch.zeros(1, 2), atol=1e-5)


@requires_torch
def test_dare_merge_drop_rate_validation() -> None:
    """dare_merge raises ValueError for drop_rate outside [0.0, 1.0)."""
    import torch
    from model_training.merging import dare_merge

    sd = [{"weight": torch.randn(4, 4)}]
    with pytest.raises(ValueError, match="drop_rate must be in"):
        dare_merge(sd, drop_rate=1.0)
    with pytest.raises(ValueError, match="drop_rate must be in"):
        dare_merge(sd, drop_rate=-0.1)


@requires_torch
def test_ties_merge_preserves_bfloat16() -> None:
    """TIES merge preserves bfloat16 dtype through float32 computation."""
    import torch
    from model_training.merging import ties_merge

    sd1 = {"weight": torch.randn(4, 4).to(torch.bfloat16)}
    sd2 = {"weight": torch.randn(4, 4).to(torch.bfloat16)}
    result = ties_merge([sd1, sd2], density=0.5)
    assert result["weight"].dtype == torch.bfloat16, (
        f"Expected bfloat16, got {result['weight'].dtype}"
    )


@requires_torch
def test_dare_merge_preserves_dtype() -> None:
    """DARE merge preserves float16 dtype through float32 computation."""
    import torch
    from model_training.merging import dare_merge

    sd1 = {"weight": torch.randn(4, 4).to(torch.float16)}
    sd2 = {"weight": torch.randn(4, 4).to(torch.float16)}
    result = dare_merge([sd1, sd2], drop_rate=0.1)
    assert result["weight"].dtype == torch.float16, (
        f"Expected float16, got {result['weight'].dtype}"
    )


@requires_torch
def test_dare_merge_seed_reproducibility() -> None:
    """DARE merge with same seed produces identical results."""
    import torch
    from model_training.merging import dare_merge

    sd1 = {"weight": torch.randn(8, 8)}
    sd2 = {"weight": torch.randn(8, 8)}

    result_a = dare_merge([sd1, sd2], drop_rate=0.3, seed=42)
    result_b = dare_merge([sd1, sd2], drop_rate=0.3, seed=42)
    assert torch.equal(result_a["weight"], result_b["weight"]), (
        "Same seed should produce identical DARE merge results"
    )

    result_c = dare_merge([sd1, sd2], drop_rate=0.3, seed=99)
    assert not torch.equal(result_a["weight"], result_c["weight"]), (
        "Different seeds should produce different DARE merge results"
    )
