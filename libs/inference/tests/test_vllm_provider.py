"""Tests for VLLMProvider with mocked AsyncOpenAI and httpx."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from inference.provider import GenerationResult, InferenceProvider
from inference.vllm_provider import VLLMProvider


def _make_openai_response(
    text: str,
    model: str,
    token_count: int = 10,
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


class TestVLLMProviderIsProvider:
    """Test that VLLMProvider satisfies the InferenceProvider contract."""

    def test_is_instance_of_inference_provider(self) -> None:
        """Test 1: VLLMProvider is an instance of InferenceProvider."""
        provider = VLLMProvider()
        assert isinstance(provider, InferenceProvider)


class TestVLLMProviderGenerate:
    """Tests for VLLMProvider.generate()."""

    async def test_generate_returns_generation_result(self) -> None:
        """Test 2: generate() returns GenerationResult with correct fields."""
        provider = VLLMProvider()
        mock_response = _make_openai_response(
            text="def hello(): pass",
            model="Qwen3.5-9B",
            token_count=12,
            finish_reason="stop",
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await provider.generate(
                prompt="write a hello function",
                model="Qwen3.5-9B",
            )

        assert isinstance(result, GenerationResult)
        assert result.text == "def hello(): pass"
        assert result.token_count == 12
        assert result.finish_reason == "stop"

    async def test_generate_with_adapter_passes_adapter_id_as_model(self) -> None:
        """Test 3: generate() with adapter_id passes adapter_id as model."""
        provider = VLLMProvider()
        mock_response = _make_openai_response(
            text="output", model="adapter-001", token_count=5
        )
        captured_calls: list[dict[str, Any]] = []

        async def capturing_create(**kwargs: Any) -> Any:
            captured_calls.append(kwargs)
            return mock_response

        with patch.object(
            provider._client.chat.completions, "create", side_effect=capturing_create
        ):
            await provider.generate(
                prompt="hello",
                model="Qwen3.5-9B",
                adapter_id="adapter-001",
            )

        assert captured_calls[0]["model"] == "adapter-001"

    async def test_generate_without_adapter_passes_model_as_is(self) -> None:
        """Test 4: generate() without adapter_id passes model parameter as-is."""
        provider = VLLMProvider()
        mock_response = _make_openai_response(text="out", model="Qwen3.5-9B")
        captured_calls: list[dict[str, Any]] = []

        async def capturing_create(**kwargs: Any) -> Any:
            captured_calls.append(kwargs)
            return mock_response

        with patch.object(
            provider._client.chat.completions, "create", side_effect=capturing_create
        ):
            await provider.generate(prompt="hello", model="Qwen3.5-9B")

        assert captured_calls[0]["model"] == "Qwen3.5-9B"


class TestVLLMProviderAdapterLifecycle:
    """Tests for VLLMProvider adapter load/unload/list operations."""

    async def test_load_adapter_posts_to_correct_endpoint(self) -> None:
        """Test 5: load_adapter() POSTs to /v1/load_lora_adapter and tracks adapter."""
        provider = VLLMProvider(base_url="http://localhost:8100/v1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_post(url: str, json: dict[str, Any]) -> Any:
            assert url == "http://localhost:8100/v1/load_lora_adapter"
            assert json == {
                "lora_name": "adapter-001",
                "lora_path": "/models/adapter-001",
            }
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_http
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.load_adapter("adapter-001", "/models/adapter-001")

        assert "adapter-001" in provider._loaded_adapters

    async def test_unload_adapter_posts_to_correct_endpoint(self) -> None:
        """Test 6: unload_adapter() POSTs to /v1/unload_lora_adapter and removes it."""
        provider = VLLMProvider(base_url="http://localhost:8100/v1")
        provider._loaded_adapters.add("adapter-001")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_post(url: str, json: dict[str, Any]) -> Any:
            assert url == "http://localhost:8100/v1/unload_lora_adapter"
            assert json == {"lora_name": "adapter-001"}
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_http
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.unload_adapter("adapter-001")

        assert "adapter-001" not in provider._loaded_adapters

    async def test_list_adapters_returns_internal_tracking_set(self) -> None:
        """Test 7: list_adapters() returns the internal tracking set contents."""
        provider = VLLMProvider()
        provider._loaded_adapters = {"adapter-001"}

        result = await provider.list_adapters()
        assert result == ["adapter-001"]

    async def test_loading_two_adapters_lists_both(self) -> None:
        """Test 8: Loading two adapters and listing returns both names (INF-06)."""
        provider = VLLMProvider(base_url="http://localhost:8100/v1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_http
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.load_adapter("adapter-001", "/models/adapter-001")
            await provider.load_adapter("adapter-002", "/models/adapter-002")

        result = await provider.list_adapters()
        assert sorted(result) == ["adapter-001", "adapter-002"]


class TestVLLMProviderAsync:
    """Tests that VLLMProvider methods are coroutine functions."""

    def test_all_methods_are_coroutine_functions(self) -> None:
        """Test 9: All methods are coroutine functions (async)."""
        provider = VLLMProvider()
        assert asyncio.iscoroutinefunction(provider.generate)
        assert asyncio.iscoroutinefunction(provider.load_adapter)
        assert asyncio.iscoroutinefunction(provider.unload_adapter)
        assert asyncio.iscoroutinefunction(provider.list_adapters)
