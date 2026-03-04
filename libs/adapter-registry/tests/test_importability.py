"""CPU-only importability smoke tests for adapter-registry."""

import pytest
from adapter_registry import (
    AdapterAlreadyExistsError,
    AdapterNotFoundError,
    AdapterRecord,
    AdapterRegistry,
)


def test_adapter_record_is_importable() -> None:
    """AdapterRecord can be imported from adapter_registry."""
    assert AdapterRecord is not None


def test_adapter_registry_is_importable() -> None:
    """AdapterRegistry can be imported from adapter_registry."""
    assert AdapterRegistry is not None


def test_exceptions_are_importable() -> None:
    """Custom exceptions can be imported from adapter_registry."""
    assert AdapterAlreadyExistsError is not None
    assert AdapterNotFoundError is not None


def test_store_raises_not_implemented() -> None:
    """AdapterRegistry.store raises NotImplementedError."""
    registry = AdapterRegistry()
    record = AdapterRecord(
        id="test-adapter-001",
        version=1,
        task_type="bug-fix",
        base_model_id="qwen2.5-coder-7b",
        rank=64,
        created_at="2026-01-01T00:00:00Z",
        file_path="/adapters/test.safetensors",
        file_hash="abc123",
        file_size_bytes=1024,
        source="distillation",
        session_id="session-001",
    )
    with pytest.raises(NotImplementedError):
        registry.store(record)


def test_retrieve_by_id_raises_not_implemented() -> None:
    """AdapterRegistry.retrieve_by_id raises NotImplementedError."""
    registry = AdapterRegistry()
    with pytest.raises(NotImplementedError):
        registry.retrieve_by_id("test-id")


def test_query_by_task_type_raises_not_implemented() -> None:
    """AdapterRegistry.query_by_task_type raises NotImplementedError."""
    registry = AdapterRegistry()
    with pytest.raises(NotImplementedError):
        registry.query_by_task_type("bug-fix")


def test_list_all_raises_not_implemented() -> None:
    """AdapterRegistry.list_all raises NotImplementedError."""
    registry = AdapterRegistry()
    with pytest.raises(NotImplementedError):
        registry.list_all()
