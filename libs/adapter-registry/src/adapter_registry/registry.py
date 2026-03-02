"""AdapterRegistry class providing CRUD operations for adapter metadata."""

from adapter_registry.exceptions import (
    AdapterAlreadyExistsError,
    AdapterNotFoundError,
)
from adapter_registry.models import AdapterRecord


class AdapterRegistry:
    """Registry for storing and querying LoRA adapter metadata.

    Provides CRUD operations backed by SQLite via SQLModel.
    All methods are currently stubs that raise NotImplementedError.

    Raises:
        AdapterAlreadyExistsError: When storing a duplicate adapter ID.
        AdapterNotFoundError: When querying for a non-existent adapter.
    """

    def store(self, record: AdapterRecord) -> None:
        """Store a new adapter record in the registry.

        Args:
            record: The AdapterRecord to persist.

        Raises:
            NotImplementedError: Method is not yet implemented.
            AdapterAlreadyExistsError: If an adapter with the same ID exists.
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
        """
        raise NotImplementedError(
            "AdapterRegistry.list_all is not yet implemented. "
            "It will return all AdapterRecords where is_archived=False."
        )
