"""AdapterRegistry class providing CRUD operations for adapter metadata."""

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, select

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

        @event.listens_for(engine, "connect")
        def _set_wal(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        SQLModel.metadata.create_all(engine)

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
                raise AdapterNotFoundError(
                    f"No adapter with id '{adapter_id}'."
                )
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
        with Session(self._engine, expire_on_commit=False) as session:
            stmt = select(AdapterRecord).where(AdapterRecord.task_type == task_type)
            results = list(session.exec(stmt).all())
            for r in results:
                session.expunge(r)
            return results

    def list_all(self) -> list[AdapterRecord]:
        """List all non-archived adapter records.

        Returns:
            List of all AdapterRecord instances where is_archived=False.

        Example:
            >>> engine = create_engine("sqlite:///adapters.db")
            >>> registry = AdapterRegistry(engine=engine)
            >>> all_adapters = registry.list_all()
        """
        with Session(self._engine, expire_on_commit=False) as session:
            stmt = select(AdapterRecord).where(
                AdapterRecord.is_archived == False  # noqa: E712
            )
            results = list(session.exec(stmt).all())
            for r in results:
                session.expunge(r)
            return results
