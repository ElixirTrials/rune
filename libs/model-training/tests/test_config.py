"""TDD wireframe tests for model_training.config module."""

import pytest

from model_training.config import get_training_config, validate_config


def test_get_training_config_raises_not_implemented() -> None:
    """get_training_config raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="get_training_config"):
        get_training_config("bug-fix")


def test_validate_config_raises_not_implemented() -> None:
    """validate_config raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="validate_config"):
        validate_config({})
