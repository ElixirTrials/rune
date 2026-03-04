"""TDD wireframe tests for inference.adapter_loader module.

Tests for get_vllm_client PASS (real implementation).
Tests for load_adapter, unload_adapter, list_loaded_adapters assert
NotImplementedError stubs (TDD red-phase contract).
The mock_vllm_client fixture from conftest.py is exercised to validate
conftest wiring per ROADMAP Phase 15 SC-3.
"""

from unittest.mock import MagicMock

import pytest
from inference.adapter_loader import (
    get_vllm_client,
    list_loaded_adapters,
    load_adapter,
    unload_adapter,
)
from openai import AsyncOpenAI


def test_get_vllm_client_returns_async_openai() -> None:
    """get_vllm_client returns an AsyncOpenAI instance with default base URL."""
    client = get_vllm_client()
    assert isinstance(client, AsyncOpenAI)


def test_get_vllm_client_custom_base_url() -> None:
    """get_vllm_client accepts a custom base_url override."""
    client = get_vllm_client("http://custom:9000/v1")
    assert str(client.base_url).rstrip("/").endswith("custom:9000/v1")


def test_load_adapter_raises_not_implemented() -> None:
    """load_adapter raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="load_adapter"):
        load_adapter("adapter-001", "Qwen/Qwen2.5-Coder-7B")


def test_load_adapter_with_mock_vllm_client(mock_vllm_client: MagicMock) -> None:
    """Exercises the mock_vllm_client conftest fixture with adapter_loader.

    Validates that the conftest.py mock_vllm_client fixture is discoverable
    and injectable into adapter_loader tests. The mock client is verified as
    a MagicMock with the expected .chat.completions.create interface, then
    load_adapter is called to confirm it still raises NotImplementedError
    (the stub hasn't been wired to the client yet).

    This satisfies ROADMAP Phase 15 SC-3: inference tests use the mock vLLM
    client fixture from the component conftest.
    """
    # Verify the mock fixture is properly configured
    assert isinstance(mock_vllm_client, MagicMock)
    assert (
        mock_vllm_client.chat.completions.create.return_value.choices[0].message.content
        == "mock response"
    )

    # load_adapter still raises — it hasn't been wired to use a client yet
    with pytest.raises(NotImplementedError, match="load_adapter"):
        load_adapter("adapter-001", "Qwen/Qwen2.5-Coder-7B")


def test_unload_adapter_raises_not_implemented() -> None:
    """unload_adapter raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="unload_adapter"):
        unload_adapter("adapter-001")


def test_list_loaded_adapters_raises_not_implemented() -> None:
    """list_loaded_adapters raises NotImplementedError with function name in message."""
    with pytest.raises(NotImplementedError, match="list_loaded_adapters"):
        list_loaded_adapters()
