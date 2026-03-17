"""AdapterRegistry class providing CRUD operations for adapter metadata."""

import json
from typing import Any

from shared.storage_utils import set_wal_mode
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, col, select

from adapter_registry.exceptions import AdapterAlreadyExistsError, AdapterNotFoundError
from adapter_registry.models import AdapterRecord


class AdapterRegistry:
    """Registry for storing and querying LoRA adapter metadata.

    Provides CRUD operations backed by SQLite via SQLModel. The registry
    is initialized with a SQLAlchemy Engine and creates tables idempotently
    on construction. WAL mode is activated automatically on every new
    SQLite connection via an event hook registered before table creation.

    Raises:
        AdapterAlreadyExistsError: When storing a duplicate adapter ID.
        AdapterNotFoundError: When querying for a non-existent adapter.

    Example:
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("sqlite:///adapters.db")
        >>> registry = AdapterRegistry(engine=engine)
        >>> registry.store(record)
    """

    def __init__(self, engine: Engine) -> None:
        """Initialize the registry with a SQLAlchemy Engine.

        Registers a WAL-mode hook on the engine before creating tables to
        ensure every connection (including those opened by create_all) uses
        Write-Ahead Logging. Table creation is idempotent — safe to call
        against an existing database.

        Args:
            engine: A SQLAlchemy Engine connected to the target SQLite database.
        """
        self._engine = engine
        set_wal_mode(engine)
        SQLModel.metadata.create_all(engine)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_query(self, stmt: Any) -> list[AdapterRecord]:
        """Execute a select statement and return detached AdapterRecord instances.

        Centralises the ``with Session ... expunge`` boilerplate used by every
        read method so it is not repeated across the class.

        Args:
            stmt: A SQLModel select statement returning AdapterRecord rows.

        Returns:
            List of AdapterRecord instances detached from the session.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            results = list(session.exec(stmt).all())
            for r in results:
                session.expunge(r)
            return results

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store(self, record: AdapterRecord) -> None:
        """Store a new adapter record in the registry.

        Args:
            record: The AdapterRecord to persist.

        Raises:
            AdapterAlreadyExistsError: If an adapter with the same ID exists.

        Example:
            >>> engine = create_engine("sqlite:///adapters.db")
            >>> registry = AdapterRegistry(engine=engine)
            >>> record = AdapterRecord(id="abc-1", version=1, task_type="bug-fix",
            ...     base_model_id="qwen2.5-coder-7b", rank=16,
            ...     created_at="2026-01-01T00:00:00Z",
            ...     file_path="/adapters/abc-1.safetensors",
            ...     file_hash="sha256hash", file_size_bytes=4096,
            ...     source="distillation", session_id="sess-001")
            >>> registry.store(record)
        """
        with Session(self._engine, expire_on_commit=False) as session:
            if session.get(AdapterRecord, record.id) is not None:
                raise AdapterAlreadyExistsError(
                    f"Adapter '{record.id}' already exists."
                )
            session.add(record)
            session.commit()
            session.expunge(record)

    def archive(self, adapter_id: str) -> None:
        """Archive an adapter by setting is_archived=True.

        Args:
            adapter_id: ID of the adapter to archive.

        Raises:
            AdapterNotFoundError: If no adapter with the given ID exists.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            record = session.get(AdapterRecord, adapter_id)
            if record is None:
                raise AdapterNotFoundError(f"No adapter with id '{adapter_id}'.")
            record.is_archived = True
            session.add(record)
            session.commit()

    def update_fitness(
        self,
        adapter_id: str,
        pass_rate: float,
        fitness_score: float,
    ) -> None:
        """Update evaluation metrics for an adapter.

        Args:
            adapter_id: ID of the adapter to update.
            pass_rate: New pass rate value.
            fitness_score: New fitness score value.

        Raises:
            AdapterNotFoundError: If no adapter with the given ID exists.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            record = session.get(AdapterRecord, adapter_id)
            if record is None:
                raise AdapterNotFoundError(f"No adapter with id '{adapter_id}'.")
            record.pass_rate = pass_rate
            record.fitness_score = fitness_score
            session.add(record)
            session.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def retrieve_by_id(self, adapter_id: str) -> AdapterRecord:
        """Retrieve a single adapter record by its unique ID.

        Args:
            adapter_id: The unique identifier of the adapter to retrieve.

        Returns:
            The matching AdapterRecord.

        Raises:
            AdapterNotFoundError: If no adapter with the given ID exists.

        Example:
            >>> engine = create_engine("sqlite:///adapters.db")
            >>> registry = AdapterRegistry(engine=engine)
            >>> record = registry.retrieve_by_id("abc-1")
        """
        with Session(self._engine, expire_on_commit=False) as session:
            record = session.get(AdapterRecord, adapter_id)
            if record is None:
                raise AdapterNotFoundError(f"No adapter with id '{adapter_id}'.")
            session.expunge(record)
            return record

    def query_by_task_type(self, task_type: str) -> list[AdapterRecord]:
        """Query all adapters matching a given task type.

        Args:
            task_type: The task category to filter by (e.g. 'bug-fix').

        Returns:
            List of matching AdapterRecord instances. Empty list if none match.

        Example:
            >>> engine = create_engine("sqlite:///adapters.db")
            >>> registry = AdapterRegistry(engine=engine)
            >>> results = registry.query_by_task_type("bug-fix")
        """
        stmt = select(AdapterRecord).where(AdapterRecord.task_type == task_type)
        return self._execute_query(stmt)

    def list_all(self) -> list[AdapterRecord]:
        """List all non-archived adapter records.

        Returns:
            List of all AdapterRecord instances where is_archived=False.

        Example:
            >>> engine = create_engine("sqlite:///adapters.db")
            >>> registry = AdapterRegistry(engine=engine)
            >>> all_adapters = registry.list_all()
        """
        stmt = select(AdapterRecord).where(
            AdapterRecord.is_archived == False  # noqa: E712
        )
        return self._execute_query(stmt)

    def query_best_for_task(
        self, task_type: str, top_k: int = 3
    ) -> list[AdapterRecord]:
        """Return the top-k highest-fitness adapters for a task type.

        Args:
            task_type: Task category to filter by.
            top_k: Maximum number of results to return.

        Returns:
            List of AdapterRecord instances ordered by fitness_score DESC.
        """
        stmt = (
            select(AdapterRecord)
            .where(
                AdapterRecord.task_type == task_type,
                AdapterRecord.is_archived == False,  # noqa: E712
            )
            .order_by(col(AdapterRecord.fitness_score).desc())
            .limit(top_k)
        )
        return self._execute_query(stmt)

    def is_task_solved(self, task_hash: str, threshold: float = 0.95) -> bool:
        """Check if any adapter for the given task hash meets the threshold.

        Args:
            task_hash: Training task hash to check.
            threshold: Minimum pass_rate to consider "solved".

        Returns:
            True if a qualifying adapter exists.
        """
        stmt = (
            select(AdapterRecord)
            .where(
                AdapterRecord.training_task_hash == task_hash,
                AdapterRecord.is_archived == False,  # noqa: E712
                col(AdapterRecord.pass_rate) >= threshold,
            )
            .limit(1)
        )
        return bool(self._execute_query(stmt))

    def get_lineage(self, adapter_id: str) -> list[str]:
        """Walk the parent_ids chain for an adapter.

        Args:
            adapter_id: Starting adapter ID.

        Returns:
            List of adapter IDs from the given adapter back to the root.
        """
        chain: list[str] = [adapter_id]
        current_id = adapter_id
        visited: set[str] = {adapter_id}

        while True:
            with Session(self._engine, expire_on_commit=False) as session:
                record = session.get(AdapterRecord, current_id)
                if record is None or record.parent_ids is None:
                    break
                parent_list: list[str] = json.loads(record.parent_ids)
                if not parent_list:
                    break
                next_id = parent_list[0]
                if next_id in visited:
                    break
                visited.add(next_id)
                chain.append(next_id)
                current_id = next_id

        return chain

    def query_unevaluated(self, task_type: str | None = None) -> list[AdapterRecord]:
        """Return adapters that have not been evaluated (pass_rate IS NULL).

        Args:
            task_type: Optional task type filter.

        Returns:
            List of unevaluated AdapterRecord instances.
        """
        stmt = select(AdapterRecord).where(
            AdapterRecord.is_archived == False,  # noqa: E712
            col(AdapterRecord.pass_rate).is_(None),  # type: ignore[arg-type]
        )
        if task_type is not None:
            stmt = stmt.where(AdapterRecord.task_type == task_type)
        return self._execute_query(stmt)

    def get_task_types(self) -> list[str]:
        """Return all distinct task types in the registry.

        Returns:
            Sorted list of unique task_type values.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            stmt = select(AdapterRecord.task_type).distinct()
            results = list(session.exec(stmt).all())
            return sorted(results)
