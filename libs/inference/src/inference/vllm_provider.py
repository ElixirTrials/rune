"""VLLMProvider: InferenceProvider implementation backed by a vLLM server.

Uses the OpenAI-compatible API for generation and vLLM's proprietary
LoRA management endpoints for hot-loading adapters at runtime.
"""

import logging
import os

import httpx
from openai import AsyncOpenAI

from inference.provider import GenerationResult, InferenceProvider

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1")

logger = logging.getLogger(__name__)


class VLLMProvider(InferenceProvider):
    """InferenceProvider backed by a vLLM server with LoRA hot-loading support.

    Communicates with vLLM via two channels:
      - AsyncOpenAI SDK for generation (OpenAI-compatible endpoint).
      - httpx for LoRA adapter management (vLLM proprietary endpoints).

    Adapter tracking is maintained in an internal set to work around vLLM
    bug #11761 (list_lora_adapters unreliable after concurrent loads).

    Attributes:
        _client: AsyncOpenAI client pointing at the vLLM server.
        _base_url: Base URL string for constructing adapter management URLs.
        _loaded_adapters: Set of currently tracked adapter IDs.

    Example:
        >>> provider = VLLMProvider(base_url="http://localhost:8100/v1")
        >>> result = await provider.generate("def hello", model="Qwen2.5-Coder-7B")
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize VLLMProvider with an AsyncOpenAI client.

        Args:
            base_url: Override URL for the vLLM server. Defaults to
                VLLM_BASE_URL env var or http://localhost:8100/v1.
        """
        self._base_url = base_url or VLLM_BASE_URL
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="not-needed-for-local-vllm",
        )
        self._loaded_adapters: set[str] = set()

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt, optionally using a loaded LoRA adapter.

        When adapter_id is provided, it is passed as the model parameter to
        the OpenAI API — this is how vLLM identifies and routes to loaded
        LoRA adapters (the adapter is referenced by its lora_name).

        Args:
            prompt: The user-facing input prompt.
            model: Base model identifier. Used as-is when no adapter is given.
            adapter_id: Name of a loaded LoRA adapter to apply. When set,
                this value replaces model in the API call.
            max_tokens: Maximum number of tokens to generate.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature override.
            top_p: Nucleus sampling threshold override.
            repetition_penalty: Repetition penalty override.

        Returns:
            GenerationResult with the generated text and metadata.

        Example:
            >>> result = await provider.generate("def fib", model="Qwen2.5-Coder-7B")
            >>> print(result.text)
        """
        effective_model = adapter_id if adapter_id is not None else model
        logger.debug(
            "generate: model=%s adapter_id=%s max_tokens=%d",
            effective_model,
            adapter_id,
            max_tokens,
        )

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = await self._client.chat.completions.create(
            model=effective_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            model=response.model,
            adapter_id=adapter_id,
            token_count=response.usage.total_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Load a LoRA adapter into the vLLM server.

        Posts to vLLM's /v1/load_lora_adapter endpoint and adds the adapter
        to the internal tracking set on success.

        Args:
            adapter_id: Unique name for the adapter (used as lora_name).
            adapter_path: Filesystem path to the adapter weights directory.

        Raises:
            httpx.HTTPStatusError: If the vLLM server returns an error response.

        Example:
            >>> await provider.load_adapter("adapter-001", "/models/adapter-001")
        """
        url = f"{self._base_url.rstrip('/')}/load_lora_adapter"
        logger.debug("load_adapter: POST %s lora_name=%s", url, adapter_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"lora_name": adapter_id, "lora_path": adapter_path},
            )
            response.raise_for_status()

        self._loaded_adapters.add(adapter_id)
        logger.info("Adapter loaded: %s", adapter_id)

    async def unload_adapter(self, adapter_id: str) -> None:
        """Unload a LoRA adapter from the vLLM server.

        Posts to vLLM's /v1/unload_lora_adapter endpoint and removes the
        adapter from the internal tracking set.

        Args:
            adapter_id: Name of the adapter to unload.

        Raises:
            httpx.HTTPStatusError: If the vLLM server returns an error response.

        Example:
            >>> await provider.unload_adapter("adapter-001")
        """
        url = f"{self._base_url.rstrip('/')}/unload_lora_adapter"
        logger.debug("unload_adapter: POST %s lora_name=%s", url, adapter_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"lora_name": adapter_id},
            )
            response.raise_for_status()

        self._loaded_adapters.discard(adapter_id)
        logger.info("Adapter unloaded: %s", adapter_id)

    async def list_adapters(self) -> list[str]:
        """List all currently loaded LoRA adapters.

        Returns the internal tracking set rather than querying vLLM to avoid
        the unreliable list endpoint (vLLM bug #11761).

        Returns:
            Sorted list of adapter IDs currently tracked as loaded.

        Example:
            >>> adapters = await provider.list_adapters()
            >>> print(adapters)  # ["adapter-001", "adapter-002"]
        """
        return sorted(self._loaded_adapters)
