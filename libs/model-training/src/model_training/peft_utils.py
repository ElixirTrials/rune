"""QLoRA PEFT configuration and adapter management.

All GPU library imports (peft, transformers, torch) are deferred inside
function bodies to ensure CPU-only importability (INFRA-05).
"""

from __future__ import annotations

from typing import Any


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

    Example:
        >>> config = build_qlora_config(rank=64, alpha=128, target_modules=["q_proj"])
    """
    from peft import LoraConfig  # deferred — GPU/peft not available in CPU CI

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def apply_lora_adapter(model: Any, config: Any) -> Any:
    """Apply a LoRA adapter to a base model.

    Args:
        model: The base model to wrap with LoRA.
        config: The LoRA configuration (from build_qlora_config).

    Returns:
        The model wrapped with a LoRA adapter via peft.get_peft_model.

    Example:
        >>> adapted_model = apply_lora_adapter(base_model, lora_config)
    """
    from peft import get_peft_model  # deferred — GPU/peft not available in CPU CI

    return get_peft_model(model, config)


def merge_adapter(model: Any) -> Any:
    """Merge LoRA weights into the base model.

    Args:
        model: A PEFT model with LoRA adapter applied.

    Returns:
        The base model with LoRA weights merged in.

    Raises:
        NotImplementedError: Adapter merging is out of scope for Phase 21.

    Example:
        >>> merged = merge_adapter(peft_model)
    """
    raise NotImplementedError(
        "merge_adapter is not yet implemented. "
        "It will merge LoRA weights into the base model and return the merged model."
    )
