"""Inference provider library for LLM generation and LoRA adapter management.

Provides a provider-agnostic interface (InferenceProvider) with vLLM and
Ollama backends, a factory for backend selection by configuration, and
structured generation results.

Provider classes (OllamaProvider, VLLMProvider, PyVLLMProvider) are lazily
imported to avoid hard failures when optional GPU packages are not installed
(e.g. in CI).

Benchmark helpers
-----------------
``inference.benchmark_backends`` contains the synchronous ``Backend`` ABC,
``VLLMBackend``, ``InferenceProviderBackend``, ``GenerationOutput``, and the
majority-vote helpers ``_majority_vote`` / ``_tally_votes``.  These bridge the
async provider API into the batched synchronous interface expected by
evaluation runners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inference.exceptions import UnsupportedOperationError
from inference.factory import get_provider, get_provider_for_step
from inference.provider import GenerationResult, InferenceProvider

if TYPE_CHECKING:
    from inference.benchmark_backends import (
        Backend,
        GenerationOutput,
        InferenceProviderBackend,
        VLLMBackend,
    )
    from inference.llamacpp_provider import LlamaCppProvider
    from inference.ollama_provider import OllamaProvider
    from inference.vllm_provider import PyVLLMProvider
    from inference.transformers_provider import TransformersProvider
    from inference.vllm_provider import VLLMProvider

__all__ = [
    # Core provider interface
    "GenerationResult",
    "InferenceProvider",
    "UnsupportedOperationError",
    # Concrete providers
    "LlamaCppProvider",
    "OllamaProvider",
    "PyVLLMProvider",
    "TransformersProvider",
    "VLLMProvider",
    # Factory
    "get_provider",
    "get_provider_for_step",
    # Benchmark backends
    "Backend",
    "GenerationOutput",
    "InferenceProviderBackend",
    "VLLMBackend",
]


def __getattr__(name: str) -> object:
    if name == "OllamaProvider":
        from inference.ollama_provider import OllamaProvider

        return OllamaProvider
    if name == "VLLMProvider":
        from inference.vllm_provider import VLLMProvider

        return VLLMProvider
    if name == "PyVLLMProvider":
        from inference.vllm_provider import PyVLLMProvider

        return PyVLLMProvider
    if name == "LlamaCppProvider":
        from inference.llamacpp_provider import LlamaCppProvider

        return LlamaCppProvider
    if name == "TransformersProvider":
        from inference.transformers_provider import TransformersProvider

        return TransformersProvider
    if name in ("Backend", "GenerationOutput", "InferenceProviderBackend", "VLLMBackend"):
        import inference.benchmark_backends as _bb

        return getattr(_bb, name)
    raise AttributeError(f"module 'inference' has no attribute {name!r}")
