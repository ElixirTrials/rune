"""Adapter loading and vLLM client for OpenAI-compatible inference.

Uses the openai package (not a direct vllm import) to communicate with
vLLM's OpenAI-compatible API endpoint. This allows the module to be
imported and instantiated on CPU-only machines without vLLM installed.
"""

import os
from typing import Any

from openai import AsyncOpenAI  # type: ignore[import-not-found]

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")


def get_vllm_client(base_url: str | None = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointing at a vLLM server.

    Uses the openai package with a custom base_url to connect to vLLM's
    OpenAI-compatible API. This avoids a direct vllm import, allowing
    instantiation on CPU-only machines.

    Args:
        base_url: Override URL for the vLLM server. Defaults to
            VLLM_BASE_URL env var or http://localhost:8000/v1.

    Returns:
        An AsyncOpenAI client configured for the vLLM endpoint.

    Example:
        >>> client = get_vllm_client()
        >>> client.base_url  # Points to vLLM server
    """
    return AsyncOpenAI(
        base_url=base_url or VLLM_BASE_URL,
        api_key="not-needed-for-local-vllm",
    )


def load_adapter(adapter_id: str, model_name: str) -> Any:
    """Load a LoRA adapter into the running vLLM server.

    Calls the vLLM dynamic LoRA loading API via the OpenAI client
    to hot-load an adapter without restarting the server.

    Args:
        adapter_id: UUID of the adapter in the registry.
        model_name: Name/path of the base model the adapter targets.

    Returns:
        Response from the vLLM adapter loading API.

    Raises:
        NotImplementedError: Method is not yet implemented.

    Example:
        >>> response = load_adapter("adapter-001", "Qwen/Qwen2.5-Coder-7B")
        # Returns vLLM API response when implemented
    """
    raise NotImplementedError(
        "load_adapter is not yet implemented. "
        "It will call the vLLM dynamic LoRA loading API via the openai client."
    )


def unload_adapter(adapter_id: str) -> Any:
    """Unload a LoRA adapter from the running vLLM server.

    Calls the vLLM dynamic LoRA unloading API via the OpenAI client
    to remove an adapter without restarting the server.

    Args:
        adapter_id: UUID of the adapter to unload.

    Returns:
        Response from the vLLM adapter unloading API.

    Raises:
        NotImplementedError: Method is not yet implemented.

    Example:
        >>> response = unload_adapter("adapter-001")
        # Returns vLLM API response when implemented
    """
    raise NotImplementedError(
        "unload_adapter is not yet implemented. "
        "It will call the vLLM dynamic LoRA unloading API via the openai client."
    )


def list_loaded_adapters() -> list[str]:
    """List all currently loaded LoRA adapters on the vLLM server.

    Queries the vLLM server for the set of LoRA adapters currently
    hot-loaded and available for inference.

    Args:
        None

    Returns:
        List of adapter IDs currently loaded on the vLLM server.

    Raises:
        NotImplementedError: Method is not yet implemented.

    Example:
        >>> adapters = list_loaded_adapters()
        # Returns ["adapter-001", "adapter-002"] when implemented
    """
    raise NotImplementedError(
        "list_loaded_adapters is not yet implemented. "
        "It will query the vLLM server for currently loaded LoRA adapters."
    )
