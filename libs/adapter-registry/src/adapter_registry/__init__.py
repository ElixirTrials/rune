"""LoRA adapter metadata registry backed by SQLite via SQLModel."""

from adapter_registry.exceptions import (
    AdapterAlreadyExistsError,
    AdapterNotFoundError,
)
from adapter_registry.models import AdapterRecord
from adapter_registry.registry import AdapterRegistry

__all__ = [
    "AdapterRecord",
    "AdapterRegistry",
    "AdapterAlreadyExistsError",
    "AdapterNotFoundError",
]
