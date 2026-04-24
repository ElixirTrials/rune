"""Pytest configuration for libs/adapter-registry.

Shared factories (make_adapter_record, etc.) are defined here for use within
this component's tests. These mirror the root conftest.py factories.
"""

from collections.abc import Generator
from typing import Any, Callable

import pytest
from adapter_registry import AdapterRegistry
from adapter_registry.models import AdapterRecord
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine


@pytest.fixture
def make_adapter_record() -> Callable[..., AdapterRecord]:
    """Create AdapterRecord instances with defaults."""

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


@pytest.fixture
def memory_engine() -> Generator[Engine, None, None]:
    """In-memory SQLite engine — disposed after each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield engine
    engine.dispose()


@pytest.fixture
def registry(memory_engine: Engine) -> AdapterRegistry:
    """AdapterRegistry backed by in-memory SQLite."""
    return AdapterRegistry(engine=memory_engine)
