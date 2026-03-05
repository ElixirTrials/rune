"""Green-phase tests for model_training.config module."""

import pytest
from model_training.config import get_training_config, validate_config


def test_get_training_config_returns_dict() -> None:
    """get_training_config returns a dict with expected keys and values."""
    result = get_training_config("bug-fix", rank=64, epochs=3)
    assert isinstance(result, dict)
    assert result["task_type"] == "bug-fix"
    assert result["rank"] == 64
    assert result["epochs"] == 3
    assert "warmup_ratio" in result
    assert result["warmup_ratio"] == 0.03
    assert result["bf16"] is True


def test_get_training_config_uses_defaults() -> None:
    """get_training_config applies default hyperparameters."""
    result = get_training_config("code-gen")
    assert result["rank"] == 64
    assert result["learning_rate"] == 2e-4
    assert result["epochs"] == 3


def test_get_training_config_alpha_is_double_rank() -> None:
    """get_training_config sets alpha to 2*rank."""
    result = get_training_config("code-gen", rank=32)
    assert result["alpha"] == 64


def test_validate_config_valid() -> None:
    """validate_config returns True for a valid configuration dict."""
    config = {
        "task_type": "bug-fix",
        "rank": 64,
        "epochs": 3,
        "learning_rate": 2e-4,
    }
    assert validate_config(config) is True


def test_validate_config_missing_key() -> None:
    """validate_config raises ValueError when required keys are missing."""
    with pytest.raises(ValueError, match="Missing required config keys"):
        validate_config({"rank": 64})


def test_validate_config_invalid_rank() -> None:
    """validate_config raises ValueError when rank is 0."""
    config = {
        "task_type": "bug-fix",
        "rank": 0,
        "epochs": 3,
        "learning_rate": 2e-4,
    }
    with pytest.raises(ValueError, match="rank"):
        validate_config(config)


def test_validate_config_rank_too_high() -> None:
    """validate_config raises ValueError when rank exceeds 256."""
    config = {
        "task_type": "bug-fix",
        "rank": 512,
        "epochs": 3,
        "learning_rate": 2e-4,
    }
    with pytest.raises(ValueError, match="rank"):
        validate_config(config)


def test_validate_config_invalid_epochs() -> None:
    """validate_config raises ValueError when epochs is 0."""
    config = {
        "task_type": "bug-fix",
        "rank": 64,
        "epochs": 0,
        "learning_rate": 2e-4,
    }
    with pytest.raises(ValueError, match="epochs"):
        validate_config(config)


def test_validate_config_invalid_learning_rate() -> None:
    """validate_config raises ValueError when learning_rate is >= 1.0."""
    config = {
        "task_type": "bug-fix",
        "rank": 64,
        "epochs": 3,
        "learning_rate": 1.5,
    }
    with pytest.raises(ValueError, match="learning_rate"):
        validate_config(config)
