"""CPU-only importability smoke tests for adapter-registry."""

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
