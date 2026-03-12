"""Shared models, utilities, and canonical prompt templates."""

from pathlib import Path

from shared.hardware import GPUInfo, HardwareBudget, HardwareProbe
from shared.rune_models import (
    AdapterRef,
    CodingSession,
    EvolMetrics,
    SwarmCheckpoint,
    SwarmConfig,
)
from shared.sandbox import (
    NsjailBackend,
    SandboxBackend,
    SandboxResult,
    SubprocessBackend,
    get_sandbox_backend,
)

__all__ = [
    "AdapterRef",
    "CodingSession",
    "EvolMetrics",
    "GPUInfo",
    "HardwareBudget",
    "HardwareProbe",
    "NsjailBackend",
    "SandboxBackend",
    "SandboxResult",
    "SubprocessBackend",
    "SwarmCheckpoint",
    "SwarmConfig",
    "get_prompts_dir",
    "get_sandbox_backend",
]


def get_prompts_dir() -> Path:
    """Return the path to the shared prompt templates directory (Jinja2 .j2 files).

    Use this as prompts_dir when calling inference.factory helpers so agents
    use the same templates with different prompt_vars (DRY). No duplicate
    template files in services.

    Returns:
        Path to libs/shared/src/shared/templates/ (package data).
    """
    return Path(__file__).resolve().parent / "templates"
