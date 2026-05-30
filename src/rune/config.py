"""Pipeline configuration dataclass and loader for Rune."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen configuration for the Rune inference and training pipeline."""

    model_id: str = "Qwen/Qwen3.5-9B"
    adapter_scaling: float = 1.0
    temperature: float = 0.3
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    thinking_budget: int = 1024
    max_phase_iterations: int = 10
    cont_multiplier: float = 1.53
    cont_budget: int = 5
    no_repeat_ngram_size: int = 12
    presence_penalty: float = 1.5
    checkpoint_path: str = ""
    seed: int | None = None
    bench: dict[str, Any] = field(default_factory=dict)
    hpo: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dictionary.

        Returns:
            All config fields as a JSON-serialisable dict.
        """
        return asdict(self)

    def save(self, path: Path) -> Path:
        """Write config as YAML to disk.

        Args:
            path: Destination file path; parent directories are created.

        Returns:
            The path written to.
        """
        import yaml  # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.to_dict(), default_flow_style=False))
        return path

    def override(self, **kwargs: Any) -> PipelineConfig:
        """Return a new config with the given fields replaced.

        Args:
            **kwargs: Field names and new values to apply.

        Returns:
            A new PipelineConfig with updated values.
        """
        d = self.to_dict()
        d.update(kwargs)
        return PipelineConfig(**d)

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Construct a config from RUNE_* environment variables.

        Returns:
            PipelineConfig with any recognised env vars applied as overrides,
            or a default instance if none are set.
        """
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
    """Load a PipelineConfig from a YAML file, or return defaults if missing.

    Args:
        path: Path to a YAML config file.

    Returns:
        Parsed PipelineConfig, or a default instance if the file does not exist.
    """
    if path.exists():
        import yaml  # noqa: PLC0415

        d = yaml.safe_load(path.read_text())
        if d is None:
            # Empty/whitespace-only file: honour the documented default fallback
            # instead of crashing on PipelineConfig(**None).
            return PipelineConfig()
        if not isinstance(d, dict):
            raise ValueError(
                f"{path} must contain a YAML mapping, got {type(d).__name__}"
            )
        return PipelineConfig(**d)
    return PipelineConfig()
