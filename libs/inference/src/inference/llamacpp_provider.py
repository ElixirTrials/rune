"""LlamaCppProvider: InferenceProvider using llama-cpp-python with LoRA support.

Loads GGUF models via llama_cpp.Llama with optional LoRA adapter paths.
Designed for Apple Silicon (Metal) local inference where adapter hot-loading
is needed — Ollama cannot load LoRA adapters, and vLLM requires a server.

IMPORTANT: llama_cpp is imported inside method bodies per INFRA-05 pattern
so that this module is importable in CPU-only CI without llama-cpp-python.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from inference.provider import GenerationResult, InferenceProvider

logger = logging.getLogger(__name__)

# Stop sequences used when generating code completions.
_STOP_SEQUENCES: list[str] = ["```\n", "\n\n\n"]


class LlamaCppProvider(InferenceProvider):
    """InferenceProvider backed by llama-cpp-python with native LoRA support.

    Unlike OllamaProvider, this loads GGUF models directly and can apply
    LoRA adapters at load time. Unlike VLLMProvider, no server is needed.

    The model is loaded lazily on first generate() call. When an adapter
    is loaded, the model is reloaded with the LoRA path applied.

    Args:
        model_path: Path to the GGUF model file.
        n_ctx: Context window size. Default: 4096.
        n_gpu_layers: Layers to offload to GPU (-1 = all). Default: -1.

    Example:
        >>> provider = LlamaCppProvider(model_path="/models/qwen2.5-coder-1.5b.gguf")
        >>> result = await provider.generate("def hello", model="ignored")
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ) -> None:
        """Initialize LlamaCppProvider with model configuration.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size. Default: 4096.
            n_gpu_layers: Layers to offload to GPU (-1 = all). Default: -1.
        """
        self._model_path = model_path or ""
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm: Any = None
        self._current_lora: str | None = None
        self._loaded_adapters: dict[str, str] = {}  # id -> path

    def _load_model_if_needed(self, lora_path: str | None = None) -> None:
        """Load or reload the GGUF model, optionally with a LoRA adapter.

        Args:
            lora_path: Path to LoRA adapter file. If different from current,
                triggers a model reload.
        """
        if self._llm is not None and lora_path == self._current_lora:
            return

        from llama_cpp import Llama  # noqa: PLC0415

        if self._llm is not None:
            del self._llm
            self._llm = None

        logger.info(
            "Loading GGUF model: %s (lora=%s, n_ctx=%d, n_gpu_layers=%d)",
            self._model_path,
            lora_path,
            self._n_ctx,
            self._n_gpu_layers,
        )

        kwargs: dict[str, Any] = {
            "model_path": self._model_path,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "verbose": False,
        }
        if lora_path:
            kwargs["lora_path"] = lora_path

        self._llm = Llama(**kwargs)
        self._current_lora = lora_path

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        """Generate text using llama-cpp-python with optional LoRA adapter.

        Args:
            prompt: The input prompt.
            model: Ignored (model is set at construction via model_path).
            adapter_id: LoRA adapter ID to apply. Must be loaded via
                load_adapter() before use.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with generated text and metadata.

        Raises:
            ValueError: If adapter_id is provided but has not been loaded.
        """
        lora_path: str | None = None
        if adapter_id:
            if adapter_id not in self._loaded_adapters:
                raise ValueError(
                    f"Adapter '{adapter_id}' has not been loaded. "
                    "Call load_adapter() first."
                )
            lora_path = self._loaded_adapters[adapter_id]

        self._load_model_if_needed(lora_path=lora_path)

        response = self._llm(  # type: ignore[union-attr]
            prompt,
            max_tokens=max_tokens,
            stop=_STOP_SEQUENCES,
        )

        text = response["choices"][0]["text"]
        token_count = response["usage"]["total_tokens"]
        finish_reason = response["choices"][0].get("finish_reason", "stop")

        return GenerationResult(
            text=text,
            model=Path(self._model_path).stem,
            adapter_id=adapter_id,
            token_count=token_count,
            finish_reason=finish_reason,
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Register a LoRA adapter for use during generation.

        The adapter is applied on next generate() call by reloading the model
        with the LoRA path. llama-cpp-python applies LoRA at model load time.

        Args:
            adapter_id: Unique name for the adapter.
            adapter_path: Filesystem path to the LoRA adapter weights (GGUF format).
        """
        self._loaded_adapters[adapter_id] = adapter_path
        logger.info("Registered adapter %s -> %s", adapter_id, adapter_path)

    async def unload_adapter(self, adapter_id: str) -> None:
        """Remove a registered LoRA adapter.

        If the currently active adapter is unloaded, the model will reload
        without it on the next generate() call.

        Args:
            adapter_id: The adapter name to remove.
        """
        if adapter_id in self._loaded_adapters:
            was_active = self._current_lora == self._loaded_adapters[adapter_id]
            del self._loaded_adapters[adapter_id]
            if was_active:
                self._current_lora = None  # Force reload without adapter
            logger.info("Unloaded adapter %s", adapter_id)

    async def list_adapters(self) -> list[str]:
        """List all registered LoRA adapter IDs.

        Returns:
            Sorted list of registered adapter IDs.
        """
        return sorted(self._loaded_adapters.keys())
