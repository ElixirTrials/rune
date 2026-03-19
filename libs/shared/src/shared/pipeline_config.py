"""Pipeline configuration for adapter scaling, generation, and prompt style.

Provides a frozen dataclass config with load/save to JSON, factory defaults,
and per-field override from environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_CONFIG_FILENAME = "pipeline_config.json"
_DEFAULT_CONFIG_DIR = Path.home() / ".rune"


@dataclass(frozen=True)
class AdapterConfig:
    """Adapter weight application settings."""

    scaling: float = 0.075
    use_bias: bool = True
    max_length: int = 2048


@dataclass(frozen=True)
class GenerationConfig:
    """LLM generation settings."""

    temperature: float = 0.3
    max_tokens: int = 1024
    repetition_penalty: float = 1.1
    top_p: float = 0.9


@dataclass(frozen=True)
class PromptConfig:
    """Prompt template selection."""

    style: str = "must_include"


@dataclass(frozen=True)
class TrajectoryConfig:
    """Trajectory template selection."""

    style: str = "full_context"


@dataclass(frozen=True)
class CalibrationConfig:
    """Per-task calibration settings."""

    enabled: bool = True
    n_trials: int = 5
    scaling_range: tuple[float, float] = (0.5, 1.5)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        d = asdict(self)
        # Convert tuple back for JSON compatibility
        d["calibration"]["scaling_range"] = list(
            d["calibration"]["scaling_range"]
        )
        return d

    def save(self, path: Path | None = None) -> Path:
        """Write config to JSON file."""
        path = path or (_DEFAULT_CONFIG_DIR / _CONFIG_FILENAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def override(self, **kwargs: Any) -> PipelineConfig:
        """Return a new config with selected fields replaced.

        Accepts dotted keys like ``adapter.scaling=0.1`` or flat
        section dicts like ``adapter={"scaling": 0.1}``.
        """
        d = self.to_dict()
        for key, value in kwargs.items():
            if "." in key:
                section, field_name = key.split(".", 1)
                d.setdefault(section, {})[field_name] = value
            elif isinstance(value, dict):
                d.setdefault(key, {}).update(value)
            else:
                d[key] = value
        return _from_dict(d)


def _from_dict(d: dict[str, Any]) -> PipelineConfig:
    """Build PipelineConfig from a plain dict."""
    cal = d.get("calibration", {})
    if "scaling_range" in cal and isinstance(cal["scaling_range"], list):
        cal["scaling_range"] = tuple(cal["scaling_range"])
    return PipelineConfig(
        adapter=AdapterConfig(**d.get("adapter", {})),
        generation=GenerationConfig(**d.get("generation", {})),
        prompt=PromptConfig(**d.get("prompt", {})),
        trajectory=TrajectoryConfig(**d.get("trajectory", {})),
        calibration=CalibrationConfig(**cal),
    )


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load config from JSON, falling back to defaults.

    Also checks ``RUNE_PIPELINE_CONFIG`` env var for the path.
    """
    if path is None:
        env_path = os.environ.get("RUNE_PIPELINE_CONFIG")
        path = Path(env_path) if env_path else _DEFAULT_CONFIG_DIR / _CONFIG_FILENAME

    if path.exists():
        d = json.loads(path.read_text())
        return _from_dict(d)
    return PipelineConfig()


def default_config() -> PipelineConfig:
    """Return the default config without reading any files."""
    return PipelineConfig()
