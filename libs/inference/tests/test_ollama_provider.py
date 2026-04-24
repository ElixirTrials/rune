"""Tests for OllamaProvider with mocked AsyncOpenAI."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inference.exceptions import UnsupportedOperationError
from inference.ollama_provider import OllamaProvider
from inference.provider import GenerationResult, InferenceProvider


def _make_openai_response(
    text: str,
    model: str,
    token_count: int = 8,
    finish_reason: str = "stop",
) -> Any:
    """Build a minimal mock chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.total_tokens = token_count

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


class TestOllamaProviderIsProvider:
    """Test that OllamaProvider satisfies the InferenceProvider contract."""

    def test_is_instance_of_inference_provider(self) -> None:
        """Test 1: OllamaProvider is an instance of InferenceProvider."""
        provider = OllamaProvider()
        assert isinstance(provider, InferenceProvider)


class TestOllamaProviderGenerate:
    """Tests for OllamaProvider.generate()."""

    async def test_generate_returns_result_with_none_adapter_id(self) -> None:
        """Test 2: generate() returns GenerationResult with adapter_id=None."""
        provider = OllamaProvider()
        mock_response = _make_openai_response(
            text="def hello(): pass",
            model="qwen3.5:9b",
            token_count=10,
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await provider.generate(
                prompt="write a hello function",
                model="qwen3.5:9b",
            )

        assert isinstance(result, GenerationResult)
        assert result.adapter_id is None
        assert result.text == "def hello(): pass"

    async def test_generate_calls_openai_with_correct_model(self) -> None:
        """Test 3: generate() calls chat.completions.create with correct model."""
        provider = OllamaProvider()
        mock_response = _make_openai_response(text="output", model="qwen3.5:9b")
        captured_calls: list[dict[str, Any]] = []

        async def capturing_create(**kwargs: Any) -> Any:
            captured_calls.append(kwargs)
            return mock_response

        with patch.object(
            provider._client.chat.completions,
            "create",
            side_effect=capturing_create,
        ):
            await provider.generate(prompt="hello", model="qwen3.5:9b")

        assert captured_calls[0]["model"] == "qwen3.5:9b"

    async def test_generate_ignores_adapter_id_and_logs_warning(self) -> None:
        """Test 8: generate() ignores adapter_id (logs warning, does not fail)."""
        provider = OllamaProvider()
        mock_response = _make_openai_response(text="output", model="qwen3.5:9b")
        captured_calls: list[dict[str, Any]] = []

        async def capturing_create(**kwargs: Any) -> Any:
            captured_calls.append(kwargs)
            return mock_response

        with patch.object(
            provider._client.chat.completions,
            "create",
            side_effect=capturing_create,
        ):
            result = await provider.generate(
                prompt="hello",
                model="qwen3.5:9b",
                adapter_id="adapter-001",
            )

        # Should still use base model, not the adapter_id
        assert captured_calls[0]["model"] == "qwen3.5:9b"
        # adapter_id is None in the result
        assert result.adapter_id is None


class TestOllamaProviderAdapterOps:
    """Tests for OllamaProvider adapter operations (raise UnsupportedOperationError)."""

    async def test_load_adapter_raises_unsupported_operation_error(self) -> None:
        """Test 4: load_adapter() raises UnsupportedOperationError."""
        provider = OllamaProvider()
        with pytest.raises(UnsupportedOperationError):
            await provider.load_adapter("adapter-001", "/models/adapter-001")

    async def test_unload_adapter_raises_unsupported_operation_error(self) -> None:
        """Test 5: unload_adapter() raises UnsupportedOperationError."""
        provider = OllamaProvider()
        with pytest.raises(UnsupportedOperationError):
            await provider.unload_adapter("adapter-001")

    async def test_list_adapters_returns_empty_list(self) -> None:
        """Test 6: list_adapters() returns empty list."""
        provider = OllamaProvider()
        result = await provider.list_adapters()
        assert result == []


class TestOllamaProviderAsync:
    """Tests that OllamaProvider methods are coroutine functions."""

    def test_all_methods_are_coroutine_functions(self) -> None:
        """Test 7: All methods are coroutine functions (async)."""
        provider = OllamaProvider()
        assert asyncio.iscoroutinefunction(provider.generate)
        assert asyncio.iscoroutinefunction(provider.load_adapter)
        assert asyncio.iscoroutinefunction(provider.unload_adapter)
        assert asyncio.iscoroutinefunction(provider.list_adapters)
