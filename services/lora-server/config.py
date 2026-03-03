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
    """

    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 1
    enable_lora: bool = True
    quantization: str = "awq"
    port: int = 8000
    health_port: int = 8001
    max_loras: int = 8

    def __post_init__(self) -> None:
        if self.tensor_parallel_size == 2:
            raise ValueError(
                "tensor_parallel_size=2 is forbidden. "
                "vLLM bug #21471: TP+LoRA causes corrupted outputs on consumer GPUs "
                "without NVLink. Use pipeline_parallel_size=2 with tensor_parallel_size=1 instead."
            )

    @classmethod
    def from_yaml(cls, path: str) -> LoraServerConfig:
        """Load configuration from a YAML file.

        Only keys matching dataclass fields are passed through;
        unknown keys in the YAML are silently ignored.
        """
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)
