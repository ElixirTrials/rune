"""Tests for model_training.peft_utils module."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from model_training.peft_utils import (
    apply_lora_adapter,
    build_qlora_config,
    merge_adapter,
)


def _gpu_available() -> bool:
    """Check if real peft is importable."""
    try:
        import peft  # noqa: F401

        return True
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="peft not available",
)


@requires_gpu
def test_build_qlora_config_returns_lora_config() -> None:
    """build_qlora_config returns a LoraConfig with the expected attributes."""
    result = build_qlora_config(rank=64, alpha=128, target_modules=["q_proj", "v_proj"])
    assert result.r == 64
    assert result.lora_alpha == 128
    assert result.task_type == "CAUSAL_LM"
    assert result.bias == "none"
    assert "embed_tokens" not in result.target_modules
    assert "lm_head" not in result.target_modules


def test_apply_lora_adapter_calls_get_peft_model() -> None:
    """apply_lora_adapter delegates to peft.get_peft_model with (model, config).

    Injects a fake 'peft' module into sys.modules so the deferred import inside
    apply_lora_adapter resolves without requiring the real peft package.
    """
    mock_model = MagicMock()
    mock_config = MagicMock()
    mock_peft_model = MagicMock()
    mock_get_peft_model = MagicMock(return_value=mock_peft_model)

    # Build a fake peft module that has get_peft_model
    fake_peft = ModuleType("peft")
    fake_peft.get_peft_model = mock_get_peft_model  # type: ignore[attr-defined]

    original = sys.modules.get("peft")
    sys.modules["peft"] = fake_peft
    try:
        result = apply_lora_adapter(mock_model, mock_config)
    finally:
        if original is None:
            del sys.modules["peft"]
        else:
            sys.modules["peft"] = original

    mock_get_peft_model.assert_called_once_with(mock_model, mock_config)
    assert result is mock_peft_model


def test_merge_adapter_raises_not_implemented() -> None:
    """merge_adapter raises NotImplementedError (out of scope for Phase 21)."""
    with pytest.raises(NotImplementedError, match="merge_adapter"):
        merge_adapter(model=None)
