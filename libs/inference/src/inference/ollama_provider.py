"""OllamaProvider: InferenceProvider implementation backed by an Ollama server.

Uses Ollama's OpenAI-compatible endpoint for generation. Adapter operations
raise UnsupportedOperationError since Ollama has no LoRA adapter concept.
"""

import logging
import os

from openai import AsyncOpenAI

from inference.exceptions import UnsupportedOperationError
from inference.provider import GenerationResult, InferenceProvider

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

logger = logging.getLogger(__name__)


class OllamaProvider(InferenceProvider):
    """InferenceProvider backed by an Ollama server.

    Uses Ollama's OpenAI-compatible API (/v1/chat/completions) for generation,
    keeping the HTTP layer symmetrical with VLLMProvider. Adapter operations
    are not supported — calling them raises UnsupportedOperationError.

    Note:
        Ollama requires a non-empty api_key but ignores its value. The string
        "ollama" is used by convention.

    Attributes:
        _client: AsyncOpenAI client pointing at the Ollama server.

    Example:
        >>> provider = OllamaProvider(base_url="http://localhost:11434/v1")
        >>> result = await provider.generate("def hello", model="qwen2.5-coder:7b")
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize OllamaProvider with an AsyncOpenAI client.

        Args:
            base_url: Override URL for the Ollama server. Defaults to
                OLLAMA_BASE_URL env var or http://localhost:11434/v1.
        """
        self._base_url = base_url or OLLAMA_BASE_URL
        # Ollama requires a non-empty api_key but ignores its value.
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="ollama",
        )

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Generate text from a prompt using the base Ollama model.

        If adapter_id is provided, a warning is logged and it is ignored —
        Ollama does not support LoRA adapters. The base model is always used.

        Args:
            prompt: The input prompt to send to the model.
            model: The Ollama model identifier (e.g. "qwen2.5-coder:7b").
            adapter_id: Ignored. If provided, a warning is logged.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            GenerationResult with adapter_id=None (Ollama has no adapter concept).

        Example:
            >>> result = await provider.generate("def fib", model="qwen2.5-coder:7b")
            >>> print(result.text)
        """
        if adapter_id is not None:
            logger.warning(
                "OllamaProvider ignoring adapter_id=%s; "
                "Ollama does not support LoRA adapters.",
                adapter_id,
            )

        logger.debug("generate: model=%s max_tokens=%d", model, max_tokens)

        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            model=response.model,
            adapter_id=None,
            token_count=response.usage.total_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Not supported by Ollama. Always raises UnsupportedOperationError.

        Args:
            adapter_id: Unused.
            adapter_path: Unused.

        Raises:
            UnsupportedOperationError: Always — Ollama does not support
                LoRA adapter loading. Use VLLMProvider for adapter operations.

        Example:
            >>> await provider.load_adapter("adapter-001", "/models/adapter-001")
            # Raises UnsupportedOperationError
        """
        raise UnsupportedOperationError(
            "OllamaProvider does not support LoRA adapter loading. "
            "Use VLLMProvider for adapter operations."
        )

    async def unload_adapter(self, adapter_id: str) -> None:
        """Not supported by Ollama. Always raises UnsupportedOperationError.

        Args:
            adapter_id: Unused.

        Raises:
            UnsupportedOperationError: Always — Ollama does not support
                LoRA adapter unloading.

        Example:
            >>> await provider.unload_adapter("adapter-001")
            # Raises UnsupportedOperationError
        """
        raise UnsupportedOperationError(
            "OllamaProvider does not support LoRA adapter unloading."
        )

    async def list_adapters(self) -> list[str]:
        """Return an empty list — Ollama has no adapter concept.

        Returns:
            Always returns an empty list.

        Example:
            >>> adapters = await provider.list_adapters()
            >>> print(adapters)  # []
        """
        return []
