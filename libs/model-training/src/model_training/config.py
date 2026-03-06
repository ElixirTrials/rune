"""Training configuration for LoRA fine-tuning.

No GPU imports required — this module is pure dict construction and validation.
"""

from __future__ import annotations

from typing import Any

_REQUIRED_KEYS = {"task_type", "rank", "epochs", "learning_rate"}


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

    Example:
        >>> config = get_training_config("bug-fix", rank=64, epochs=3)
        >>> config["task_type"]
        'bug-fix'
    """
    return {
        "task_type": task_type,
        "rank": rank,
        "alpha": 2 * rank,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "save_strategy": "no",
        "logging_steps": 1,
        "report_to": "none",
        "eval_strategy": "no",
        "target_modules": ["q_proj", "v_proj"],
        "dropout": 0.1,
    }


def validate_config(config: dict[str, Any]) -> bool:
    """Validate training configuration fields and value ranges.

    Checks that all required fields are present and their values
    fall within acceptable ranges.

    Args:
        config: A training configuration dict to validate.

    Returns:
        True if the configuration is valid.

    Raises:
        ValueError: If required keys are missing or values are out of range.

    Example:
        >>> valid = validate_config({"task_type": "bug-fix", "rank": 64,
        ...                         "epochs": 3, "learning_rate": 2e-4})
        >>> valid
        True
    """
    missing = _REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    rank = config["rank"]
    if not (isinstance(rank, int) and rank > 0 and rank <= 256):
        raise ValueError(f"rank must be an integer in range (0, 256], got {rank!r}")

    epochs = config["epochs"]
    if not (isinstance(epochs, int) and epochs > 0 and epochs <= 100):
        raise ValueError(f"epochs must be an integer in range (0, 100], got {epochs!r}")

    lr = config["learning_rate"]
    if not (isinstance(lr, float) and lr > 0.0 and lr < 1.0):
        raise ValueError(f"learning_rate must be a float in range (0, 1), got {lr!r}")

    return True
