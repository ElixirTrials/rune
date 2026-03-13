"""Inference provider library for LLM generation and LoRA adapter management.

Provides a provider-agnostic interface (InferenceProvider) with vLLM and
Ollama backends, a factory for backend selection by configuration, and
structured generation results.

Provider classes (OllamaProvider, VLLMProvider) are lazily imported to avoid
hard failures when the ``openai`` package is not installed (e.g. in CI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inference.exceptions import UnsupportedOperationError
from inference.factory import get_provider, get_provider_for_step
from inference.provider import GenerationResult, InferenceProvider

if TYPE_CHECKING:
    from inference.llamacpp_provider import LlamaCppProvider
    from inference.ollama_provider import OllamaProvider
    from inference.transformers_provider import TransformersProvider
    from inference.vllm_provider import VLLMProvider

__all__ = [
    "GenerationResult",
    "InferenceProvider",
    "LlamaCppProvider",
    "OllamaProvider",
    "TransformersProvider",
    "UnsupportedOperationError",
    "VLLMProvider",
    "get_provider",
    "get_provider_for_step",
]


def __getattr__(name: str) -> object:
    if name == "OllamaProvider":
        from inference.ollama_provider import OllamaProvider

        return OllamaProvider
    if name == "VLLMProvider":
        from inference.vllm_provider import VLLMProvider

        return VLLMProvider
    if name == "LlamaCppProvider":
        from inference.llamacpp_provider import LlamaCppProvider

        return LlamaCppProvider
    if name == "TransformersProvider":
        from inference.transformers_provider import TransformersProvider

        return TransformersProvider
    raise AttributeError(f"module 'inference' has no attribute {name!r}")
