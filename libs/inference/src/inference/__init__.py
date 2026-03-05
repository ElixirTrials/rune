"""Inference provider library for LLM generation and LoRA adapter management.

Provides a provider-agnostic interface (InferenceProvider) with vLLM and
Ollama backends, a factory for backend selection by configuration, and
structured generation results.
"""

from inference.exceptions import UnsupportedOperationError
from inference.factory import get_provider, get_provider_for_step
from inference.ollama_provider import OllamaProvider
from inference.provider import GenerationResult, InferenceProvider
from inference.vllm_provider import VLLMProvider

__all__ = [
    "GenerationResult",
    "InferenceProvider",
    "OllamaProvider",
    "UnsupportedOperationError",
    "VLLMProvider",
    "get_provider",
    "get_provider_for_step",
]
