"""Tests for AdapterRegistry CRUD methods.

These tests verify the full implementation: constructor, WAL mode, and all
four CRUD methods with their expected behaviors including error cases.
"""

import pytest
from adapter_registry import AdapterRegistry
from adapter_registry.exceptions import (
    AdapterAlreadyExistsError,
    AdapterNotFoundError,
)
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@pytest.fixture
def engine(tmp_path) -> Engine:
    """Create an in-memory SQLite engine for test isolation."""
    db_path = tmp_path / "test_registry.db"
    return create_engine(f"sqlite:///{db_path}")


@pytest.fixture
def registry(engine) -> AdapterRegistry:
    """Create an AdapterRegistry instance with the test engine."""
    return AdapterRegistry(engine=engine)


# --- Constructor ---


def test_constructor_requires_engine(engine) -> None:
    """AdapterRegistry accepts an Engine parameter and creates tables."""
    registry = AdapterRegistry(engine=engine)
    assert registry is not None


# --- store() ---


def test_store_persists_record(registry, make_adapter_record) -> None:
    """store() persists a record that can be retrieved by id."""
    record = make_adapter_record()
    registry.store(record)
    retrieved = registry.retrieve_by_id(record.id)
    assert retrieved.id == record.id


def test_store_raises_on_duplicate_id(registry, make_adapter_record) -> None:
    """store() raises AdapterAlreadyExistsError on duplicate ID."""
    record = make_adapter_record()
    registry.store(record)
    with pytest.raises(AdapterAlreadyExistsError):
        registry.store(make_adapter_record())  # same id="test-adapter-001"


def test_store_returns_none(registry, make_adapter_record) -> None:
    """store() returns None on success."""
    record = make_adapter_record()
    result = registry.store(record)
    assert result is None


# --- retrieve_by_id() ---


def test_retrieve_by_id_returns_matching_record(registry, make_adapter_record) -> None:
    """retrieve_by_id() returns the stored AdapterRecord."""
    record = make_adapter_record(id="unique-adapter-xyz")
    registry.store(record)
    retrieved = registry.retrieve_by_id("unique-adapter-xyz")
    assert retrieved.id == "unique-adapter-xyz"
    assert retrieved.task_type == record.task_type


def test_retrieve_by_id_raises_when_missing(registry) -> None:
    """retrieve_by_id() raises AdapterNotFoundError for unknown ID."""
    with pytest.raises(AdapterNotFoundError):
        registry.retrieve_by_id("nonexistent-id")


# --- query_by_task_type() ---


def test_query_by_task_type_returns_matching_records(
    registry, make_adapter_record
) -> None:
    """query_by_task_type() returns only records with the specified task_type."""
    registry.store(make_adapter_record(id="a1", task_type="bug-fix"))
    registry.store(make_adapter_record(id="a2", task_type="feature"))
    registry.store(make_adapter_record(id="a3", task_type="bug-fix"))

    results = registry.query_by_task_type("bug-fix")
    assert len(results) == 2
    assert all(r.task_type == "bug-fix" for r in results)


def test_query_by_task_type_returns_empty_list_when_none(registry) -> None:
    """query_by_task_type() returns empty list when no records match."""
    results = registry.query_by_task_type("nonexistent-type")
    assert results == []


# --- list_all() ---


def test_list_all_excludes_archived_records(registry, make_adapter_record) -> None:
    """list_all() excludes records where is_archived=True."""
    registry.store(make_adapter_record(id="active-1", is_archived=False))
    registry.store(make_adapter_record(id="archived-1", is_archived=True))
    registry.store(make_adapter_record(id="active-2", is_archived=False))

    results = registry.list_all()
    ids = [r.id for r in results]
    assert "active-1" in ids
    assert "active-2" in ids
    assert "archived-1" not in ids


def test_list_all_returns_only_non_archived(registry, make_adapter_record) -> None:
    """list_all() returns list[AdapterRecord] with is_archived=False."""
    registry.store(make_adapter_record(id="r1", is_archived=False))
    registry.store(make_adapter_record(id="r2", is_archived=True))

    results = registry.list_all()
    assert len(results) == 1
    assert results[0].id == "r1"


def test_list_all_returns_empty_when_all_archived(
    registry, make_adapter_record
) -> None:
    """list_all() returns empty list when all records are archived."""
    registry.store(make_adapter_record(id="archived", is_archived=True))
    results = registry.list_all()
    assert results == []


# --- WAL mode integration ---


def test_wal_mode_enabled(tmp_path) -> None:
    """AdapterRegistry enables WAL journal mode on file-based SQLite engine."""
    from sqlalchemy import text

    db_path = tmp_path / "test.db"
    wal_engine = create_engine(f"sqlite:///{db_path}")
    AdapterRegistry(engine=wal_engine)
    with wal_engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert result == "wal"


# --- Concurrent writes integration ---


def test_concurrent_writes_no_deadlock(tmp_path, make_adapter_record) -> None:
    """Five concurrent writes complete without deadlock or errors."""
    import threading

    db_path = tmp_path / "concurrent.db"
    conc_engine = create_engine(f"sqlite:///{db_path}")
    conc_registry = AdapterRegistry(engine=conc_engine)
    errors: list[Exception] = []

    def write(adapter_id: str) -> None:
        try:
            conc_registry.store(make_adapter_record(id=adapter_id))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write, args=(f"t-{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent writes produced errors: {errors}"
    assert len(conc_registry.list_all()) == 5
