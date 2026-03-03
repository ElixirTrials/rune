"""VLLMClient stub wrapping OpenAI's AsyncOpenAI for vLLM inference.

Uses the OpenAI-compatible API exposed by vLLM rather than importing
vLLM directly, keeping the client lightweight and decoupled from GPU deps.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI  # type: ignore[import-not-found]

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")


class VLLMClient:
    """Async client for vLLM's OpenAI-compatible API.

    Wraps AsyncOpenAI to communicate with the local vLLM server.
    Methods are stubs that will be implemented when the vLLM dynamic
    LoRA loading API is integrated.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url or VLLM_BASE_URL,
            api_key="not-needed-for-local-vllm",
        )

    async def load_adapter(self, adapter_id: str, model_name: str) -> Any:
        """Load a LoRA adapter into the running vLLM server.

        Will use vLLM's dynamic LoRA loading API to hot-swap adapters
        without restarting the server.

        Args:
            adapter_id: Identifier for the LoRA adapter to load.
            model_name: Base model the adapter was trained on.

        Raises:
            NotImplementedError: Pending vLLM dynamic LoRA loading API integration.
        """
        raise NotImplementedError(
            f"load_adapter('{adapter_id}', '{model_name}') is not yet implemented. "
            "Requires vLLM dynamic LoRA loading API integration -- "
            "see vLLM docs on /v1/load_lora_adapter endpoint."
        )

    async def generate(
        self, prompt: str, model: str, adapter_id: str | None = None
    ) -> Any:
        """Generate a completion using vLLM with an optional LoRA adapter.

        Will call the OpenAI-compatible chat completions endpoint on the
        local vLLM server, optionally routing through a loaded LoRA adapter.

        Args:
            prompt: The input prompt for generation.
            model: The base model identifier.
            adapter_id: Optional LoRA adapter to use for generation.

        Raises:
            NotImplementedError: Pending chat completion call implementation.
        """
        raise NotImplementedError(
            f"generate(model='{model}', adapter_id='{adapter_id}') is not yet implemented. "
            "Will use self._client.chat.completions.create() with the vLLM server."
        )
