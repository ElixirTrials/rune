"""Pipeline configuration dataclass and loader for Rune."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Single source of truth for the base model. The instruct variant is required so
# the pre-warmed Sakana doc-to-lora adapter (warm start) is compatible. Override
# per-process with the RUNE_BASE_MODEL env var or repo-root config.yaml.
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen configuration for the Rune inference and training pipeline."""

    model_id: str = DEFAULT_MODEL_ID
    # --- Model generation profile (model-specific; set WITH the model id) ---
    # These encode one model's generation contract. The defaults below are for
    # Qwen3-4B-Instruct-2507 (a NON-thinking instruct model). Switching models
    # means switching this profile — see config.yaml. Hardcoding any of these in
    # the runner is what caused the thinking-phase degeneration (#52): the forced
    # </think> thinking phase is wrong for a non-thinking model, and a flat
    # presence_penalty on code tokens drives single-word collapses.
    thinking_budget: int = 0  # 0 => non-thinking path (required for *-Instruct)
    presence_penalty: float = 0.0  # flat presence penalty harms code; 0 for codegen
    dtype: str = "bfloat16"  # torch dtype name; getattr(torch, dtype)
    attn_implementation: str = "flash_attention_2"
    # --- Inference / generation (task-level, not model-specific) ---
    adapter_scaling: float = 1.0
    temperature: float = 0.3
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    top_p: float = 0.9
    max_phase_iterations: int = 10
    cont_multiplier: float = 1.53
    cont_budget: int = 5
    no_repeat_ngram_size: int = 12
    # Prompt regime: "full" = spec in the prompt (default). "reference_a"/
    # "reference_b" = spec lives ONLY in the adapter conditioning, prompt refers
    # to the mission by name (issue #52 spec-in-adapter test). _a = plain ## Task
    # (faithful to c3 training); _b = Mission/Specification/Definition of Done +
    # signature in ## Current Code. Empty sections are omitted from the adapter.
    prompt_mode: str = "full"
    # In-loop correctness judge: after a candidate passes the spec's public
    # example, the model is asked for a specific failing input; a grounded verdict
    # flips the result to failing so diagnose->repair engages on held-out-style
    # logic bugs the single public example misses.
    model_judge: bool = True
    judge_temperature: float = 0.2
    judge_max_tokens: int = 256
    # Repair-signal experiment flags (issue52 repair-signal A/B).
    repair_brief_enabled: bool = True
    plan_gate_enabled: bool = True
    replan_on_complexity: bool = True
    max_repairs: int = 4
    plan_gate_max_attempts: int = 2
    # --- Engine oracle / retention / ship gate (issue #52) ---
    # Requirement kinds that route repair but do not demote best_code or block ship.
    advisory_requirement_kinds: tuple[str, ...] = ("constraint_scale",)
    # Rank visible-correct-but-slow (constraint_scale-only) as quality 3.
    constraint_scale_pass_quality: bool = True
    # Empirical complexity probe (big_o) when Constraints imply large inputs.
    complexity_probe_min_n: int = 8
    complexity_probe_max_n: int = 1200
    complexity_probe_n_repeats: int = 3
    complexity_probe_per_run_timeout_s: float = 5.0
    # Wall-clock budget for empirical big_o; on timeout, hotswap complexity adapter.
    complexity_empirical_timeout_s: float = 15.0
    complexity_judge_enabled: bool = True
    complexity_judge_temperature: float = 0.1
    complexity_judge_max_tokens: int = 384
    # Union doctest examples from the task spec into wired public_checks.
    merge_spec_public_checks: bool = True
    # When no candidate passes public checks, ship best retained attempt anyway.
    ship_best_on_exhaustion: bool = True
    ship_best_min_quality: int = 1
    # Complexity repair brief: preserve algorithm, optimize in place (vs replan).
    complexity_repair_preserve_logic: bool = True
    # --- Adapter (hypernetwork) profile ---
    checkpoint_path: str = ""  # trained hypernet checkpoint driving the adapter
    warmstart_checkpoint: str = ""  # warm-start (e.g. Sakana doc-to-lora) provenance
    seed: int | None = None
    bench: dict[str, Any] = field(default_factory=dict)
    hpo: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dictionary.

        Returns:
            All config fields as a JSON-serialisable dict.
        """
        d = asdict(self)
        d["advisory_requirement_kinds"] = list(d["advisory_requirement_kinds"])
        return d

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

    @staticmethod
    def _env_overrides() -> dict[str, Any]:
        """Collect field overrides from recognised RUNE_* environment variables."""
        overrides: dict[str, Any] = {}
        env_map: dict[str, tuple[str, type]] = {
            "RUNE_BASE_MODEL": ("model_id", str),
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
        return overrides

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Construct a config from RUNE_* environment variables.

        Returns:
            PipelineConfig with any recognised env vars applied as overrides,
            or a default instance if none are set.
        """
        overrides = cls._env_overrides()
        return cls(**overrides) if overrides else cls()


def _repo_config_path() -> Path:
    """Resolve the canonical config.yaml: RUNE_CONFIG env, else repo-root file."""
    env = os.environ.get("RUNE_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config.yaml"


def load_rune_config(path: Path | None = None) -> PipelineConfig:
    """Return the single source of truth for Rune settings.

    Resolution order (later wins): dataclass defaults -> config.yaml -> RUNE_*
    env overrides. This is what tools, scripts, and CLI commands should call
    instead of hardcoding a model id or any other setting. Env overrides apply
    whether or not an explicit path is given, so e.g. RUNE_BASE_MODEL wins
    uniformly.

    Args:
        path: Config YAML to read. Defaults to the repo-root config.yaml
            (or RUNE_CONFIG).

    Returns:
        The merged PipelineConfig.
    """
    cfg = load_config(path if path is not None else _repo_config_path())
    overrides = PipelineConfig._env_overrides()
    return cfg.override(**overrides) if overrides else cfg


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
        # `training:` is the D2LTrainConfig surface (read by load_train_config);
        # the inference/engine PipelineConfig ignores it so one config.yaml can
        # hold both.
        d = {k: v for k, v in d.items() if k != "training"}
        kinds = d.get("advisory_requirement_kinds")
        if isinstance(kinds, list):
            d["advisory_requirement_kinds"] = tuple(kinds)
        return PipelineConfig(**d)
    return PipelineConfig()
