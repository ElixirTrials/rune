"""LoRA server configuration with safety constraints for multi-GPU inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraServerConfig:
    """Configuration for the vLLM-based LoRA inference server.

    Enforces PP=2/TP=1 layout to avoid vLLM bug #21471 where TP+LoRA
    produces corrupted outputs on consumer GPUs without NVLink.

    Attributes:
        model: HuggingFace model identifier.
        pipeline_parallel_size: Number of pipeline parallel stages.
        tensor_parallel_size: Number of tensor parallel shards.
        enable_lora: Whether to enable LoRA adapter support.
        quantization: Quantization method (e.g. 'awq').
        port: Port for the vLLM inference server.
        health_port: Port for the health sidecar.
        max_loras: Maximum number of concurrent LoRA adapters.
        max_lora_rank: Maximum rank for LoRA adapters (capped at 64 to avoid OOM).
        gpu_memory_utilization: Fraction of GPU VRAM for the model (0.80 leaves
            headroom for LoRA).

    Example:
        >>> config = LoraServerConfig()
        >>> config.pipeline_parallel_size
        2
        >>> config.tensor_parallel_size
        1
    """

    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 1
    enable_lora: bool = True
    quantization: str = "awq"
    port: int = 8000
    health_port: int = 8001
    max_loras: int = 8
    max_lora_rank: int = 64
    gpu_memory_utilization: float = 0.80

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization."""
        if self.tensor_parallel_size == 2:
            raise ValueError(
                "tensor_parallel_size=2 is forbidden. "
                "vLLM bug #21471: TP+LoRA causes corrupted "
                "outputs on consumer GPUs without NVLink. "
                "Use pipeline_parallel_size=2 with "
                "tensor_parallel_size=1 instead."
            )

    @classmethod
    def from_yaml(cls, path: str) -> LoraServerConfig:
        """Load configuration from a YAML file.

        Only keys matching dataclass fields are passed through;
        unknown keys in the YAML are silently ignored.

        Args:
            path: Filesystem path to the YAML config file.

        Returns:
            A LoraServerConfig populated from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.

        Example:
            >>> config = LoraServerConfig.from_yaml("config.yml")
            >>> config.model
            'Qwen/Qwen2.5-Coder-7B-Instruct'
        """
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)
