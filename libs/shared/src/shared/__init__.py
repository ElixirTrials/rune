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
from shared.storage_utils import create_service_engine, set_wal_mode

__all__ = [
    "AdapterRef",
    "CodingSession",
    "EvolMetrics",
    "GPUInfo",
    "HardwareBudget",
    "HardwareProbe",
    "MIN_TRAINING_VRAM_MB",
    "NsjailBackend",
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
    "set_wal_mode",
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
