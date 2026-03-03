"""TDD wireframe tests for AdapterRegistry CRUD methods.

These tests assert the TDD red-phase contract: each CRUD method raises
NotImplementedError with the method name in the message. Tests use factory
fixtures from root conftest.py.
"""

import pytest

from adapter_registry import AdapterRegistry


def test_store_raises_not_implemented(make_adapter_record) -> None:
    """AdapterRegistry.store raises NotImplementedError with method name."""
    registry = AdapterRegistry()
    record = make_adapter_record()
    with pytest.raises(NotImplementedError, match="store"):
        registry.store(record)


def test_retrieve_by_id_raises_not_implemented() -> None:
    """AdapterRegistry.retrieve_by_id raises NotImplementedError with method name."""
    registry = AdapterRegistry()
    with pytest.raises(NotImplementedError, match="retrieve_by_id"):
        registry.retrieve_by_id("test-adapter-001")


def test_query_by_task_type_raises_not_implemented() -> None:
    """AdapterRegistry.query_by_task_type raises NotImplementedError with method name."""
    registry = AdapterRegistry()
    with pytest.raises(NotImplementedError, match="query_by_task_type"):
        registry.query_by_task_type("bug-fix")


def test_list_all_raises_not_implemented() -> None:
    """AdapterRegistry.list_all raises NotImplementedError with method name."""
    registry = AdapterRegistry()
    with pytest.raises(NotImplementedError, match="list_all"):
        registry.list_all()
