"""Custom exceptions for the inference library."""


class UnsupportedOperationError(Exception):
    """Raised when a provider does not support the requested operation.

    Used primarily by OllamaProvider to signal that LoRA adapter operations
    are not available for Ollama-based inference.

    Example:
        >>> raise UnsupportedOperationError("OllamaProvider does not support adapters.")
    """
