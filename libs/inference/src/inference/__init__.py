"""Inference library for vLLM-based code generation and adapter management.

Provides functions for managing LoRA adapters on a running vLLM server
and generating code completions via the OpenAI-compatible API.
"""

from inference.adapter_loader import (
    get_vllm_client,
    list_loaded_adapters,
    load_adapter,
    unload_adapter,
)
from inference.completion import (
    batch_generate,
    generate_completion,
    generate_with_adapter,
)

__all__ = [
    "batch_generate",
    "generate_completion",
    "generate_with_adapter",
    "get_vllm_client",
    "list_loaded_adapters",
    "load_adapter",
    "unload_adapter",
]
