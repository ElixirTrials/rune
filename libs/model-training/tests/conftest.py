"""Pytest configuration for libs/model-training.

Shared factories are provided by the root conftest.py
and available here automatically via pytest fixture discovery.
"""

# Pre-import the HuggingFace ``datasets`` package at conftest collection time
# so each xdist worker has it fully resolved before the first test runs.
# Without this, parallel workers can race during ``from datasets import Dataset``
# inside a test body and surface a spurious ``ImportError: cannot import name
# 'Dataset' from 'datasets' (unknown location)`` — a known interaction between
# datasets' lazy submodule loader and xdist's process-spawn timing.
try:  # pragma: no cover - import warm-up only
    import datasets as _datasets  # noqa: F401

    _ = _datasets.Dataset  # force attribute resolution
except Exception:  # noqa: BLE001 — datasets may legitimately be missing
    pass

from typing import Any, Callable

import pytest
from adapter_registry.models import AdapterRecord


@pytest.fixture
def make_adapter_record() -> Callable[..., AdapterRecord]:
    """Create AdapterRecord instances with defaults (mirrors root conftest)."""

    def _factory(**kwargs: Any) -> AdapterRecord:
        defaults: dict[str, Any] = {
            "id": "test-adapter-001",
            "version": 1,
            "task_type": "bug-fix",
            "base_model_id": "Qwen/Qwen3.5-9B",
            "rank": 16,
            "created_at": "2026-01-01T00:00:00Z",
            "file_path": "/adapters/test-adapter-001.safetensors",
            "file_hash": "abc123def456",
            "file_size_bytes": 1024,
            "pass_rate": None,
            "fitness_score": None,
            "source": "distillation",
            "session_id": "test-session-001",
            "is_archived": False,
            "parent_ids": None,
            "generation": 0,
            "training_task_hash": None,
            "agent_id": None,
        }
        return AdapterRecord(**{**defaults, **kwargs})

    return _factory
