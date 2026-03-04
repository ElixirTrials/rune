"""AdapterRegistry class providing CRUD operations for adapter metadata."""

from adapter_registry.models import AdapterRecord


class AdapterRegistry:
    """Registry for storing and querying LoRA adapter metadata.

    Provides CRUD operations backed by SQLite via SQLModel.
    All methods are currently stubs that raise NotImplementedError.

    Raises:
        AdapterAlreadyExistsError: When storing a duplicate adapter ID.
        AdapterNotFoundError: When querying for a non-existent adapter.

    Example:
        >>> registry = AdapterRegistry()
        >>> registry.store(record)  # Persists adapter when implemented
    """

    def store(self, record: AdapterRecord) -> None:
        """Store a new adapter record in the registry.

        Args:
            record: The AdapterRecord to persist.

        Raises:
            NotImplementedError: Method is not yet implemented.
            AdapterAlreadyExistsError: If an adapter with the same ID exists.

        Example:
            >>> registry = AdapterRegistry()
            >>> record = AdapterRecord(id="abc-1", version=1, task_type="bug-fix",
            ...     base_model_id="qwen2.5-coder-7b", rank=16,
            ...     created_at="2026-01-01T00:00:00Z",
            ...     file_path="/adapters/abc-1.safetensors",
            ...     file_hash="sha256hash", file_size_bytes=4096,
            ...     source="distillation", session_id="sess-001")
            >>> registry.store(record)  # Persists to SQLite when implemented
        """
        raise NotImplementedError(
            "AdapterRegistry.store is not yet implemented. "
            "It will insert the AdapterRecord into SQLite and verify ID uniqueness."
        )

    def retrieve_by_id(self, adapter_id: str) -> AdapterRecord:
        """Retrieve a single adapter record by its unique ID.

        Args:
            adapter_id: The unique identifier of the adapter to retrieve.

        Returns:
            The matching AdapterRecord.

        Raises:
            NotImplementedError: Method is not yet implemented.
            AdapterNotFoundError: If no adapter with the given ID exists.

        Example:
            >>> registry = AdapterRegistry()
            >>> record = registry.retrieve_by_id("abc-1")
        """
        raise NotImplementedError(
            "AdapterRegistry.retrieve_by_id is not yet implemented. "
            "It will query SQLite for the adapter with the given ID."
        )

    def query_by_task_type(self, task_type: str) -> list[AdapterRecord]:
        """Query all adapters matching a given task type.

        Args:
            task_type: The task category to filter by (e.g. 'bug-fix').

        Returns:
            List of matching AdapterRecord instances.

        Raises:
            NotImplementedError: Method is not yet implemented.

        Example:
            >>> registry = AdapterRegistry()
            >>> results = registry.query_by_task_type("bug-fix")
        """
        raise NotImplementedError(
            "AdapterRegistry.query_by_task_type is not yet implemented. "
            "It will query SQLite filtered by task_type index."
        )

    def list_all(self) -> list[AdapterRecord]:
        """List all non-archived adapter records.

        Returns:
            List of all AdapterRecord instances where is_archived=False.

        Raises:
            NotImplementedError: Method is not yet implemented.

        Example:
            >>> registry = AdapterRegistry()
            >>> all_adapters = registry.list_all()
        """
        raise NotImplementedError(
            "AdapterRegistry.list_all is not yet implemented. "
            "It will return all AdapterRecords where is_archived=False."
        )
