"""Abstract base class and shared types for inference providers.

Defines the provider-agnostic API that the agent loop consumes. Concrete
implementations (VLLMProvider, OllamaProvider) fulfil this interface for
their respective backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Structured result returned by InferenceProvider.generate().

    Attributes:
        text: The generated text output from the model.
        model: The model identifier used for generation.
        adapter_id: The LoRA adapter applied during generation, or None
            if no adapter was used.
        token_count: Total number of tokens consumed (prompt + completion).
        finish_reason: Reason generation stopped (e.g. "stop", "length").

    Example:
        >>> result = GenerationResult(
        ...     text="def hello(): pass",
        ...     model="Qwen/Qwen2.5-Coder-7B",
        ...     adapter_id=None,
        ...     token_count=10,
        ...     finish_reason="stop",
        ... )
    """

    text: str
    model: str
    adapter_id: str | None
    token_count: int
    finish_reason: str


class InferenceProvider(ABC):
    """Abstract base class for inference providers.

    Defines a provider-agnostic API for text generation and LoRA adapter
    lifecycle management. All methods are async because every provider
    communicates over HTTP.

    Concrete implementations:
        - VLLMProvider: Full LoRA support via vLLM's dynamic loading API.
        - OllamaProvider: Base-model inference only; adapter ops raise
          UnsupportedOperationError.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt to send to the model.
            model: The model identifier to use for generation.
            adapter_id: Optional LoRA adapter to apply during generation.
                If None, uses the base model directly.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A GenerationResult containing the generated text and metadata.

        Example:
            >>> result = await provider.generate("def hello", model="Qwen2.5-Coder-7B")
            >>> print(result.text)
        """
        ...

    @abstractmethod
    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Load a LoRA adapter into the inference server.

        Args:
            adapter_id: Unique name for the adapter (used as lora_name in vLLM).
            adapter_path: Filesystem path to the adapter weights directory.

        Raises:
            UnsupportedOperationError: If the provider does not support adapters.
            httpx.HTTPStatusError: If the server returns an error response.

        Example:
            >>> await provider.load_adapter("adapter-001", "/models/adapter-001")
        """
        ...

    @abstractmethod
    async def unload_adapter(self, adapter_id: str) -> None:
        """Unload a previously loaded LoRA adapter.

        Args:
            adapter_id: The adapter name to remove from the server.

        Raises:
            UnsupportedOperationError: If the provider does not support adapters.
            httpx.HTTPStatusError: If the server returns an error response.

        Example:
            >>> await provider.unload_adapter("adapter-001")
        """
        ...

    @abstractmethod
    async def list_adapters(self) -> list[str]:
        """List all currently loaded LoRA adapters.

        Returns:
            Sorted list of adapter IDs currently available for inference.
            Returns an empty list if no adapters are loaded or the provider
            does not support adapters.

        Example:
            >>> adapters = await provider.list_adapters()
            >>> print(adapters)  # ["adapter-001", "adapter-002"]
        """
        ...
