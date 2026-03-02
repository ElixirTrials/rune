"""Custom exceptions for adapter registry operations."""


class AdapterRegistryError(Exception):
    """Base exception for adapter registry operations."""


class AdapterAlreadyExistsError(AdapterRegistryError):
    """Raised when attempting to store an adapter with a duplicate ID."""


class AdapterNotFoundError(AdapterRegistryError):
    """Raised when an adapter matching the query criteria does not exist."""
