"""Model registry for configurable LoRA fine-tuning targets.

Provides ModelConfig (frozen Pydantic model) and ModelRegistry with named
presets for each supported base model. No GPU imports — safe to import anywhere.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ModelConfig",
    "ModelRegistry",
    "validate_against_probe",
]


class ModelConfig(BaseModel, frozen=True):
    """Immutable configuration for a base model + training defaults.

    Attributes:
        canonical_name: Short lookup key (e.g. "qwen3.5-9b").
        model_id: HuggingFace repository ID.
        warm_start_adapter_id: Pre-trained PEFT adapter to continue from.
        default_lora_rank: LoRA rank for fresh initialization.
        default_lora_alpha: LoRA alpha scaling factor.
        attn_implementation: Attention implementation override (e.g. "eager").
        expected_num_layers: Expected total layer count for probe validation.
        expected_hidden_size: Expected hidden dimension for probe validation.
        gradient_accumulation_steps: Training gradient accumulation default.
        lr_scheduler_type: Learning rate scheduler type.
        epochs: Number of training epochs.
        quirks: Model-specific overrides and notes.
    """

    canonical_name: str
    model_id: str
    warm_start_adapter_id: str | None = None
    default_lora_rank: int = 64
    default_lora_alpha: int = 32
    attn_implementation: str | None = None
    expected_num_layers: int = 32
    expected_hidden_size: int = 4096
    gradient_accumulation_steps: int = 16
    lr_scheduler_type: str = "constant"
    epochs: int = 3
    quirks: dict[str, Any] = Field(default_factory=dict)


class ModelRegistry:
    """Registry of named ModelConfig presets.

    Usage:
        registry = ModelRegistry.default()
        config = registry.get("qwen3.5-9b")
    """

    _default_instance: ClassVar[ModelRegistry | None] = None

    def __init__(self) -> None:
        """Initialize an empty model registry."""
        self._presets: dict[str, ModelConfig] = {}

    def register(self, config: ModelConfig) -> None:
        """Register a model configuration preset.

        Args:
            config: ModelConfig to register under its canonical_name.
        """
        self._presets[config.canonical_name] = config

    def get(self, name: str) -> ModelConfig:
        """Look up a model configuration by canonical name.

        Args:
            name: Canonical model name (e.g. "qwen3.5-9b").

        Returns:
            The registered ModelConfig.

        Raises:
            KeyError: If no preset exists for the given name.
        """
        if name not in self._presets:
            available = ", ".join(sorted(self._presets.keys()))
            raise KeyError(
                f"No model config registered for {name!r}. Available: {available}"
            )
        return self._presets[name]

    def list_names(self) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(self._presets.keys())

    @classmethod
    def default(cls) -> ModelRegistry:
        """Return singleton registry with built-in presets.

        Presets:
            - qwen3.5-9b: Qwen3.5-9B with DeltaCoder warm-start
            - qwen3-coder-next: Qwen3-Coder-Next (hybrid attention)
        """
        if cls._default_instance is not None:
            return cls._default_instance

        registry = cls()

        registry.register(
            ModelConfig(
                canonical_name="qwen3.5-9b",
                model_id="Qwen/Qwen3.5-9B",
                warm_start_adapter_id=("danielcherubini/Qwen3.5-DeltaCoder-9B"),
                default_lora_rank=64,
                default_lora_alpha=32,
                attn_implementation="eager",
                expected_num_layers=32,
                expected_hidden_size=4096,
                gradient_accumulation_steps=16,
                lr_scheduler_type="constant",
                epochs=3,
                quirks={
                    "flash_attention_warning": (
                        "Flash + sample packing causes loss collapse"
                    ),
                },
            )
        )

        registry.register(
            ModelConfig(
                canonical_name="qwen3-coder-next",
                model_id="Qwen/Qwen3-Coder-Next",
                warm_start_adapter_id=None,
                default_lora_rank=8,
                default_lora_alpha=16,
                attn_implementation=None,
                expected_num_layers=48,
                expected_hidden_size=2048,
                gradient_accumulation_steps=4,
                lr_scheduler_type="cosine",
                epochs=3,
            )
        )

        cls._default_instance = registry
        return registry


def validate_against_probe(
    config: ModelConfig, probe_result: dict[str, Any]
) -> list[str]:
    """Validate a ModelConfig against probe results.

    Checks expected_num_layers against discovered attention layer count and
    expected_hidden_size against probed feature dimensions.

    Args:
        config: ModelConfig with expected architecture dimensions.
        probe_result: Output from probe_model() or loaded probe cache.

    Returns:
        List of warning messages (empty if all checks pass).
    """
    warnings: list[str] = []

    layer_indices = probe_result.get("attention_layer_indices", [])
    if layer_indices:
        max_layer = max(layer_indices)
        if max_layer >= config.expected_num_layers:
            msg = (
                f"Probe found layer index {max_layer} but "
                f"{config.canonical_name} expects {config.expected_num_layers} "
                f"total layers"
            )
            warnings.append(msg)
            logger.warning(msg)

    feature_sizes = probe_result.get("feature_sizes", {})
    for proj_name, dims in feature_sizes.items():
        in_dim = dims.get("in", 0)
        if in_dim and in_dim != config.expected_hidden_size:
            # Only warn for input projections (q/k/v) where in == hidden_size
            if proj_name in ("q_proj", "k_proj", "v_proj"):
                msg = (
                    f"{proj_name} input dim {in_dim} != expected "
                    f"hidden_size {config.expected_hidden_size} for "
                    f"{config.canonical_name}"
                )
                warnings.append(msg)
                logger.warning(msg)

    if not warnings:
        logger.info("Probe validation passed for %s", config.canonical_name)

    return warnings
