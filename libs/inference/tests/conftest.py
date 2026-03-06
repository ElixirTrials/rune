"""Pytest configuration for libs/inference.

Provides an autouse fixture that clears the provider instance cache between
tests, preventing state leaks across test cases.
"""

import pytest


@pytest.fixture(autouse=True)
def clear_provider_cache() -> None:  # type: ignore[return]
    """Clear the provider factory cache before and after each test.

    Prevents provider instances cached in one test from appearing in another,
    which would break cache-identity and cache-key isolation tests.
    """
    from inference import factory

    factory._provider_cache.clear()
    yield
    factory._provider_cache.clear()
