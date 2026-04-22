"""Pytest configuration for libs/model-training.

Shared factories are provided by the root conftest.py
and available here automatically via pytest fixture discovery.
"""

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
            "base_model_id": "Qwen/Qwen2.5-Coder-7B",
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
