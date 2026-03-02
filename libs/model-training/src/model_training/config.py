"""Training configuration stubs for LoRA fine-tuning.

All functions raise NotImplementedError. No GPU imports required.
"""

from __future__ import annotations

from typing import Any, Optional


def get_training_config(
    task_type: str,
    rank: int = 64,
    epochs: int = 3,
    learning_rate: float = 2e-4,
) -> dict[str, Any]:
    """Return a training configuration dict with hyperparameters.

    Generates a configuration appropriate for the given task type,
    with sensible defaults for QLoRA fine-tuning.

    Args:
        task_type: Task category (e.g. 'bug-fix', 'feature-impl').
        rank: LoRA rank for the adapter.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        A dict containing all training hyperparameters.

    Raises:
        NotImplementedError: Method is not yet implemented.
    """
    raise NotImplementedError(
        "get_training_config is not yet implemented. "
        "It will return a training configuration dict with hyperparameters "
        "for the given task type."
    )


def validate_config(config: dict[str, Any]) -> bool:
    """Validate training configuration fields and value ranges.

    Checks that all required fields are present and their values
    fall within acceptable ranges.

    Args:
        config: A training configuration dict to validate.

    Returns:
        True if the configuration is valid.

    Raises:
        NotImplementedError: Method is not yet implemented.
    """
    raise NotImplementedError(
        "validate_config is not yet implemented. "
        "It will validate training configuration fields and value ranges."
    )
