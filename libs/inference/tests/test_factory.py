"""Tests for inference.factory — provider factory with instance cache.

Tests cover:
  - Provider type dispatch (vllm, ollama, env var default, unknown)
  - Instance caching by (provider_type, base_url) key
  - get_provider_for_step() delegation from step config dict
"""

import pytest
from inference.factory import get_provider, get_provider_for_step
from inference.ollama_provider import OllamaProvider
from inference.vllm_provider import VLLMProvider


def test_get_provider_vllm_returns_vllm_provider() -> None:
    """get_provider('vllm') returns a VLLMProvider instance."""
    provider = get_provider("vllm")
    assert isinstance(provider, VLLMProvider)


def test_get_provider_ollama_returns_ollama_provider() -> None:
    """get_provider('ollama') returns an OllamaProvider instance."""
    provider = get_provider("ollama")
    assert isinstance(provider, OllamaProvider)


def test_get_provider_uses_env_var_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_provider() with no args reads INFERENCE_PROVIDER env var, defaulting to vllm."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    provider = get_provider()
    assert isinstance(provider, OllamaProvider)


def test_get_provider_env_var_default_is_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_provider() with no args defaults to vllm when env var is unset."""
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    provider = get_provider()
    assert isinstance(provider, VLLMProvider)


def test_get_provider_unknown_raises_value_error() -> None:
    """get_provider('unknown') raises ValueError with a descriptive message."""
    with pytest.raises(ValueError, match="unknown"):
        get_provider("unknown")


def test_get_provider_caches_same_instance() -> None:
    """Calling get_provider('vllm') twice returns the same object (cache hit)."""
    provider_a = get_provider("vllm")
    provider_b = get_provider("vllm")
    assert provider_a is provider_b


def test_get_provider_different_base_url_gives_different_instance() -> None:
    """Different base_urls produce different cached instances."""
    provider_a = get_provider("vllm", base_url="http://localhost:8100/v1")
    provider_c = get_provider("vllm", base_url="http://other-host:8100/v1")
    assert provider_a is not provider_c


def test_get_provider_for_step_returns_ollama_provider() -> None:
    """get_provider_for_step({'provider': 'ollama'}) returns OllamaProvider."""
    provider = get_provider_for_step({"provider": "ollama"})
    assert isinstance(provider, OllamaProvider)


def test_get_provider_for_step_custom_base_url() -> None:
    """get_provider_for_step with base_url passes the URL to the provider."""
    provider = get_provider_for_step(
        {"provider": "vllm", "base_url": "http://custom:8100/v1"}
    )
    assert isinstance(provider, VLLMProvider)
    # The instance is cached under the custom URL key — verify it is the same
    # object when called again with the same config
    provider2 = get_provider_for_step(
        {"provider": "vllm", "base_url": "http://custom:8100/v1"}
    )
    assert provider is provider2
