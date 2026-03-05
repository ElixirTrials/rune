"""Provider factory with instance cache for the inference library.

Selects between VLLMProvider and OllamaProvider based on configuration,
caching instances by (provider_type, base_url) to avoid redundant construction.
"""

import os

from inference.provider import InferenceProvider

# Module-level constants for default values only; actual env var reads happen
# at call time inside get_provider() so monkeypatch.setenv() works in tests.
_DEFAULT_INFERENCE_PROVIDER = "vllm"
_DEFAULT_VLLM_BASE_URL = "http://localhost:8100/v1"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

_provider_cache: dict[tuple[str, str], InferenceProvider] = {}


def _clear_cache() -> None:
    """Clear the provider instance cache.

    Called by the clear_provider_cache conftest fixture to prevent cache
    state leaking between tests.
    """
    _provider_cache.clear()


def get_provider(
    provider_type: str | None = None,
    base_url: str | None = None,
) -> InferenceProvider:
    """Return a cached InferenceProvider for the given backend.

    Resolves the provider type from the argument or the INFERENCE_PROVIDER
    env var (default: "vllm"). Resolves the base URL from the argument or
    the per-backend env var (VLLM_BASE_URL / OLLAMA_BASE_URL). Instances
    are cached by the (provider_type, base_url) tuple so repeated calls
    with the same arguments return the identical object.

    Args:
        provider_type: One of "vllm" or "ollama". If None, falls back to
            the INFERENCE_PROVIDER environment variable (default "vllm").
        base_url: Override URL for the backend server. If None, the
            per-backend default env var is used.

    Returns:
        A cached InferenceProvider instance for the requested backend.

    Raises:
        ValueError: If provider_type is not "vllm" or "ollama".

    Example:
        >>> provider = get_provider("vllm")
        >>> isinstance(provider, VLLMProvider)
        True
    """
    ptype = (
        provider_type
        or os.environ.get("INFERENCE_PROVIDER", _DEFAULT_INFERENCE_PROVIDER)
    ).lower()

    resolved_url: str
    if ptype == "vllm":
        resolved_url = (
            base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_VLLM_BASE_URL)
        )
    elif ptype == "ollama":
        resolved_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        )
    else:
        raise ValueError(
            f"Unknown provider type: '{ptype}'. "
            "Supported values: 'vllm', 'ollama'."
        )

    cache_key = (ptype, resolved_url)
    if cache_key not in _provider_cache:
        if ptype == "vllm":
            from inference.vllm_provider import VLLMProvider

            _provider_cache[cache_key] = VLLMProvider(base_url=resolved_url)
        else:
            from inference.ollama_provider import OllamaProvider

            _provider_cache[cache_key] = OllamaProvider(base_url=resolved_url)

    return _provider_cache[cache_key]


def get_provider_for_step(step_config: dict[str, str]) -> InferenceProvider:
    """Return a cached InferenceProvider configured from a step config dict.

    Reads "provider" and optionally "base_url" from the step config and
    delegates to get_provider(). Designed for use by the agent loop where
    each pipeline step may specify its own provider and server URL.

    Args:
        step_config: Dict with optional keys:
            - "provider": Provider type ("vllm" or "ollama").
            - "base_url": Override URL for the backend server.

    Returns:
        A cached InferenceProvider instance for the step's backend.

    Raises:
        ValueError: If the provider type in step_config is not supported.

    Example:
        >>> provider = get_provider_for_step({"provider": "ollama"})
        >>> isinstance(provider, OllamaProvider)
        True
    """
    return get_provider(
        provider_type=step_config.get("provider"),
        base_url=step_config.get("base_url"),
    )
