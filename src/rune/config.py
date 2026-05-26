"""Pipeline configuration dataclass and loader for Rune."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen configuration for the Rune inference and training pipeline.

    Attributes:
        model_id: HuggingFace model identifier.
        adapter_scaling: LoRA adapter weight scaling factor.
        temperature: Sampling temperature for freeform generation.
        max_tokens: Maximum new tokens per generation call.
        repetition_penalty: Repetition penalty applied during generation.
        top_p: Nucleus sampling probability cutoff.
        thinking_budget: Max tokens allocated to chain-of-thought thinking.
        phase_max_tokens: Per-phase token overrides keyed by phase name.
        max_phase_iterations: Maximum iterations before a phase is abandoned.
        prompt_style: Template style identifier (e.g. "skeleton").
        trajectory_style: Trajectory serialisation style (e.g. "prose").
        adapter_ttl_days: Days before an adapter is eligible for pruning.
        checkpoint_path: Path to the hypernetwork checkpoint file.
    """

    model_id: str = "Qwen/Qwen3.5-9B"
    adapter_scaling: float = 1.0
    temperature: float = 0.3
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    thinking_budget: int = 1024
    phase_max_tokens: dict[str, int] = field(default_factory=dict)
    max_phase_iterations: int = 10
    prompt_style: str = "skeleton"
    trajectory_style: str = "prose"
    adapter_ttl_days: int = 7
    checkpoint_path: str = ""
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
        return PipelineConfig(**d)
    return PipelineConfig()
