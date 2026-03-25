"""Shared models, utilities, and canonical prompt templates."""

from pathlib import Path

from shared.hardware import (
    MIN_TRAINING_VRAM_MB,
    VRAM_PER_LORA_MB,
    GPUInfo,
    HardwareBudget,
    HardwareProbe,
    get_best_device,
)
from shared.rune_models import (
    AdapterRef,
    CodingSession,
    EvolMetrics,
    PipelinePhase,
    SwarmCheckpoint,
    SwarmConfig,
    TaskStatus,
)
from shared.sandbox import (
    NsjailBackend,
    SandboxBackend,
    SandboxResult,
    SubprocessBackend,
    get_sandbox_backend,
)
from shared.template_loader import render_prompt, render_trajectory

# storage_utils is intentionally NOT imported at module level (INFRA-05 pattern):
# it pulls sqlalchemy/sqlmodel which are heavy DB deps not needed by lightweight
# consumers like the benchmark runner or inference provider.  Import on demand.

__all__ = [
    "AdapterRef",
    "CodingSession",
    "EvolMetrics",
    "GPUInfo",
    "HardwareBudget",
    "HardwareProbe",
    "MIN_TRAINING_VRAM_MB",
    "NsjailBackend",
    "PipelinePhase",
    "SandboxBackend",
    "SandboxResult",
    "SubprocessBackend",
    "SwarmCheckpoint",
    "SwarmConfig",
    "TaskStatus",
    "VRAM_PER_LORA_MB",
    "create_service_engine",
    "get_best_device",
    "get_prompts_dir",
    "get_sandbox_backend",
    "render_prompt",
    "render_trajectory",
    "set_wal_mode",
]


def __getattr__(name: str) -> object:
    """Lazy-load heavy DB symbols to avoid sqlalchemy import at startup."""
    if name in ("create_service_engine", "set_wal_mode"):
        from shared.storage_utils import create_service_engine, set_wal_mode  # noqa: PLC0415

        globals()["create_service_engine"] = create_service_engine
        globals()["set_wal_mode"] = set_wal_mode
        return globals()[name]
    raise AttributeError(f"module 'shared' has no attribute {name!r}")


def get_prompts_dir() -> Path:
    """Return the path to the shared prompt templates directory (Jinja2 .j2 files).

    Use this as prompts_dir when calling inference.factory helpers so agents
    use the same templates with different prompt_vars (DRY). No duplicate
    template files in services.

    Returns:
        Path to libs/shared/src/shared/templates/ (package data).
    """
    return Path(__file__).resolve().parent / "templates"
