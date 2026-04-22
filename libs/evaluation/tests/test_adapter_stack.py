"""Tests for adapter_stack.load_adapter_stack()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from evaluation.benchmarks.adapter_stack import AdapterStack, load_adapter_stack


def _make_mock_registry(records: list[MagicMock]) -> MagicMock:
    """Return a mock AdapterRegistry that returns records from retrieve_by_id."""
    registry = MagicMock()
    registry.retrieve_by_id.side_effect = lambda aid: next(
        (r for r in records if r.id == aid), None
    )
    return registry


def test_load_adapter_stack_empty_adapter_ids() -> None:
    """load_adapter_stack with no adapter_ids returns an AdapterStack with empty list."""
    provider = MagicMock()
    stack = load_adapter_stack(
        base_model="Qwen/Qwen3.5-9B",
        adapter_ids=[],
        provider=provider,
        registry=MagicMock(),
    )
    assert isinstance(stack, AdapterStack)
    assert stack.base_model == "Qwen/Qwen3.5-9B"
    assert stack.adapter_ids == []


def test_load_adapter_stack_with_adapters() -> None:
    """load_adapter_stack with adapter_ids resolves file_paths from registry."""
    record = MagicMock()
    record.id = "adapter-001"
    record.file_path = "/adapters/adapter-001"
    registry = _make_mock_registry([record])
    provider = MagicMock()

    stack = load_adapter_stack(
        base_model="Qwen/Qwen3.5-9B",
        adapter_ids=["adapter-001"],
        provider=provider,
        registry=registry,
    )
    assert stack.adapter_ids == ["adapter-001"]
    assert stack.adapter_paths == {"adapter-001": "/adapters/adapter-001"}


def test_load_adapter_stack_missing_adapter_raises() -> None:
    """load_adapter_stack raises ValueError for unknown adapter_id."""
    from adapter_registry.exceptions import AdapterNotFoundError

    registry = MagicMock()
    registry.retrieve_by_id.side_effect = AdapterNotFoundError("not found")
    with pytest.raises(ValueError, match="adapter-999"):
        load_adapter_stack(
            base_model="Qwen/Qwen3.5-9B",
            adapter_ids=["adapter-999"],
            provider=MagicMock(),
            registry=registry,
        )


def test_adapter_stack_repr() -> None:
    """AdapterStack has a useful repr."""
    stack = AdapterStack(
        base_model="Qwen/Qwen3.5-9B",
        adapter_ids=["a1", "a2"],
        adapter_paths={"a1": "/p1", "a2": "/p2"},
        provider=MagicMock(),
    )
    r = repr(stack)
    assert "Qwen" in r
    assert "a1" in r


def test_adapter_stack_describe() -> None:
    """AdapterStack.describe() returns a dict with base_model and adapter_ids."""
    stack = AdapterStack(
        base_model="base",
        adapter_ids=["x"],
        adapter_paths={"x": "/p"},
        provider=MagicMock(),
    )
    d = stack.describe()
    assert d["base_model"] == "base"
    assert d["adapter_ids"] == ["x"]
