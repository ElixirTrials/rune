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
    max_tokens: int = 2048
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
class DecomposeConfig:
    """Decompose phase settings."""

    skip_threshold: int = 200


@dataclass(frozen=True)
class ReasoningLoopConfig:
    """Adapter-compressed reasoning loop settings."""

    max_turns: int = 20
    context_budget_ratio: float = 0.75
    sliding_window_tokens: int = 1024
    chunk_threshold: int = 1024
    enable_chunk_composition: bool = False
    code_scaling_boost: float = 1.2
    default_merge_method: str = "ties"
    collapse_cosine_threshold: float = 0.95
    collapse_norm_min: float = 0.1
    collapse_norm_max: float = 10.0
    collapse_repetition_threshold: float = 0.8
    adapter_target_modules: tuple[str, ...] | None = None
    adapter_layer_selection: str = "all"
    phase_sliding_windows: dict[str, int] = field(default_factory=lambda: {
        "decompose": 256,
        "plan": 512,
        "code": 1024,
        "code_repair": 1536,
        "integrate": 2048,
        "diagnose": 512,
    })


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decompose: DecomposeConfig = field(default_factory=DecomposeConfig)
    reasoning_loop: ReasoningLoopConfig = field(default_factory=ReasoningLoopConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        d = asdict(self)
        # Convert tuple back for JSON compatibility
        d["calibration"]["scaling_range"] = list(d["calibration"]["scaling_range"])
        rl_d = d.get("reasoning_loop", {})
        if rl_d.get("adapter_target_modules") is not None:
            rl_d["adapter_target_modules"] = list(rl_d["adapter_target_modules"])
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
    rl = d.get("reasoning_loop", {})
    if "adapter_target_modules" in rl and isinstance(
        rl["adapter_target_modules"], list
    ):
        rl["adapter_target_modules"] = tuple(rl["adapter_target_modules"])
    if "phase_sliding_windows" not in rl:
        rl["phase_sliding_windows"] = dict(ReasoningLoopConfig().phase_sliding_windows)
    return PipelineConfig(
        adapter=AdapterConfig(**d.get("adapter", {})),
        generation=GenerationConfig(**d.get("generation", {})),
        prompt=PromptConfig(**d.get("prompt", {})),
        trajectory=TrajectoryConfig(**d.get("trajectory", {})),
        calibration=CalibrationConfig(**cal),
        decompose=DecomposeConfig(**d.get("decompose", {})),
        reasoning_loop=ReasoningLoopConfig(**rl),
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


def resolve_reasoning_loop_config(base: ReasoningLoopConfig) -> ReasoningLoopConfig:
    """Apply env var overrides to a ReasoningLoopConfig."""
    overrides: dict[str, Any] = {}

    env_map: dict[str, tuple[str, Any]] = {
        "RUNE_MAX_REASONING_TURNS": ("max_turns", int),
        "RUNE_CONTEXT_BUDGET_RATIO": ("context_budget_ratio", float),
        "RUNE_SLIDING_WINDOW_TOKENS": ("sliding_window_tokens", int),
        "RUNE_CHUNK_THRESHOLD": ("chunk_threshold", int),
        "RUNE_ENABLE_CHUNK_COMPOSITION": (
            "enable_chunk_composition",
            lambda v: v.lower() in ("true", "1", "yes"),
        ),
        "RUNE_CODE_SCALING_BOOST": ("code_scaling_boost", float),
        "RUNE_MERGE_METHOD": ("default_merge_method", str),
        "RUNE_COLLAPSE_COSINE_THRESHOLD": ("collapse_cosine_threshold", float),
        "RUNE_COLLAPSE_NORM_MIN": ("collapse_norm_min", float),
        "RUNE_COLLAPSE_NORM_MAX": ("collapse_norm_max", float),
        "RUNE_COLLAPSE_REPETITION_THRESHOLD": ("collapse_repetition_threshold", float),
        "RUNE_ADAPTER_TARGET_MODULES": (
            "adapter_target_modules",
            lambda v: tuple(v.split(",")),
        ),
        "RUNE_ADAPTER_LAYER_SELECTION": ("adapter_layer_selection", str),
    }

    for env_key, (field_name, converter) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            overrides[field_name] = converter(val)

    phase_windows = dict(base.phase_sliding_windows)
    for phase in phase_windows:
        env_val = os.environ.get(f"RUNE_SLIDING_WINDOW_{phase.upper()}")
        if env_val is not None:
            phase_windows[phase] = int(env_val)
    overrides["phase_sliding_windows"] = phase_windows

    no_real_overrides = overrides == {
        "phase_sliding_windows": base.phase_sliding_windows
    }
    if not overrides or no_real_overrides:
        return base

    d = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    d.update(overrides)
    return ReasoningLoopConfig(**d)
