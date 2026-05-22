from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    model_id: str = "Qwen/Qwen3.5-9B"
    adapter_scaling: float = 0.075
    temperature: float = 0.3
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    thinking_budget: int = 1024
    phase_max_tokens: dict[str, int] = field(default_factory=dict)
    max_phase_iterations: int = 5
    prompt_style: str = "skeleton"
    trajectory_style: str = "prose"
    adapter_ttl_days: int = 7
    checkpoint_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def override(self, **kwargs: Any) -> PipelineConfig:
        d = self.to_dict()
        d.update(kwargs)
        return PipelineConfig(**d)

    @classmethod
    def from_env(cls) -> PipelineConfig:
        overrides: dict[str, Any] = {}
        env_map: dict[str, tuple[str, type]] = {
            "RUNE_TEMPERATURE": ("temperature", float),
            "RUNE_MAX_TOKENS": ("max_tokens", int),
            "RUNE_REPETITION_PENALTY": ("repetition_penalty", float),
            "RUNE_TOP_P": ("top_p", float),
            "RUNE_THINKING_BUDGET": ("thinking_budget", int),
            "RUNE_MAX_PHASE_ITERATIONS": ("max_phase_iterations", int),
            "RUNE_ADAPTER_SCALING": ("adapter_scaling", float),
        }
        for env_key, (field_name, converter) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                overrides[field_name] = converter(val)
        if not overrides:
            return cls()
        return cls(**overrides)


def load_config(path: Path) -> PipelineConfig:
    if path.exists():
        d = json.loads(path.read_text())
        return PipelineConfig(**d)
    return PipelineConfig()
