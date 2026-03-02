"""Lazy singleton decorator for expensive initializations."""

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Registry of all singleton instances for test cleanup
_singleton_registry: list[Callable[[], None]] = []


def lazy_singleton(func: Callable[[], T]) -> Callable[[], T]:
    """Decorator to lazily evaluate and cache a function result (singleton).

    Thread-safe implementation using a lock to prevent race conditions
    in concurrent environments.
    """
    _instance: Any = None
    _lock = threading.Lock()

    def reset() -> None:
        nonlocal _instance
        with _lock:
            _instance = None

    def wrapper() -> T:
        nonlocal _instance
        if _instance is None:
            with _lock:
                # Double-checked locking pattern
                if _instance is None:
                    _instance = func()
        return _instance

    # Register this singleton's reset function
    _singleton_registry.append(reset)

    return wrapper


def _clear_all_singletons() -> None:
    """Reset all registered singletons. Used for test isolation."""
    for reset_fn in _singleton_registry:
        reset_fn()
