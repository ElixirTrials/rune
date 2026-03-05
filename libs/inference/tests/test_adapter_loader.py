"""Smoke tests verifying inference provider imports work after refactor.

Replaces old adapter_loader TDD stub tests. Real provider tests are in
test_vllm_provider.py and test_ollama_provider.py.
"""

from inference import (
    GenerationResult,
    InferenceProvider,
    VLLMProvider,
    get_provider,
)


def test_inference_exports_provider_types() -> None:
    """Verify key provider types are importable from inference package."""
    assert InferenceProvider is not None
    assert GenerationResult is not None
    assert VLLMProvider is not None
    assert get_provider is not None
