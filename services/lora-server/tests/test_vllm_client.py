"""Smoke test verifying VLLMProvider is importable from inference lib.

VLLMClient has been absorbed into VLLMProvider in libs/inference/.
Full provider tests live in libs/inference/tests/test_vllm_provider.py.
"""

from inference.vllm_provider import VLLMProvider


def test_vllm_provider_importable_from_inference() -> None:
    """VLLMProvider is importable as the replacement for VLLMClient."""
    provider = VLLMProvider(base_url="http://localhost:8100/v1")
    assert provider is not None
