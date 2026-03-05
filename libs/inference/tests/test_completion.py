"""Smoke tests verifying inference provider exports after refactor.

Replaces old completion.py TDD stub tests. Real provider tests are in
test_vllm_provider.py and test_ollama_provider.py.
"""

from inference import (
    OllamaProvider,
    UnsupportedOperationError,
    get_provider_for_step,
)


def test_inference_exports_ollama_and_factory() -> None:
    """Verify OllamaProvider and factory are importable from inference package."""
    assert OllamaProvider is not None
    assert UnsupportedOperationError is not None
    assert get_provider_for_step is not None
