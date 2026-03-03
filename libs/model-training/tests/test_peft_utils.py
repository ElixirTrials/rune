"""TDD wireframe tests for model_training.peft_utils module."""

import pytest

from model_training.peft_utils import apply_lora_adapter, build_qlora_config, merge_adapter


def test_build_qlora_config_raises_not_implemented() -> None:
    """build_qlora_config raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="build_qlora_config"):
        build_qlora_config(rank=64, alpha=128, target_modules=["q_proj"])


def test_apply_lora_adapter_raises_not_implemented() -> None:
    """apply_lora_adapter raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="apply_lora_adapter"):
        apply_lora_adapter(model=None, config=None)


def test_merge_adapter_raises_not_implemented() -> None:
    """merge_adapter raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="merge_adapter"):
        merge_adapter(model=None)
