"""QLoRA PEFT configuration and adapter management stubs.

All functions raise NotImplementedError. GPU library imports are
deferred behind TYPE_CHECKING guards to ensure CPU-only importability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # Future: from peft import LoraConfig


def build_qlora_config(
    rank: int,
    alpha: int,
    target_modules: list[str],
    dropout: float = 0.1,
) -> Any:
    """Build a QLoRA configuration for PEFT fine-tuning.

    Args:
        rank: LoRA rank (dimensionality of low-rank matrices).
        alpha: LoRA alpha scaling factor.
        target_modules: List of module names to apply LoRA to.
        dropout: Dropout probability for LoRA layers.

    Returns:
        A peft LoraConfig instance configured for QLoRA.

    Raises:
        NotImplementedError: Method is not yet implemented.
    """
    raise NotImplementedError(
        "build_qlora_config is not yet implemented. "
        "It will instantiate peft.LoraConfig with QLoRA quantization settings."
    )


def apply_lora_adapter(model: Any, config: Any) -> Any:
    """Apply a LoRA adapter to a base model.

    Args:
        model: The base model to wrap with LoRA.
        config: The LoRA configuration (from build_qlora_config).

    Returns:
        The model wrapped with a LoRA adapter via peft.get_peft_model.

    Raises:
        NotImplementedError: Method is not yet implemented.
    """
    raise NotImplementedError(
        "apply_lora_adapter is not yet implemented. "
        "It will wrap the base model with a LoRA adapter using peft.get_peft_model."
    )


def merge_adapter(model: Any) -> Any:
    """Merge LoRA weights into the base model.

    Args:
        model: A PEFT model with LoRA adapter applied.

    Returns:
        The base model with LoRA weights merged in.

    Raises:
        NotImplementedError: Method is not yet implemented.
    """
    raise NotImplementedError(
        "merge_adapter is not yet implemented. "
        "It will merge LoRA weights into the base model and return the merged model."
    )
