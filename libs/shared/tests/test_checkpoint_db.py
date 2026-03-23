"""Tests for shared.checkpoint_db — SwarmCheckpointDB."""

import threading

from shared.checkpoint_db import SwarmCheckpointDB
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine


def _make_db() -> SwarmCheckpointDB:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return SwarmCheckpointDB(engine)


def test_mark_running_then_completed() -> None:
    db = _make_db()
    db.mark_running("task-1", "agent-1")
    assert db.is_completed("task-1") is False
    db.mark_completed("task-1", "success")
    assert db.is_completed("task-1") is True


def test_is_completed_false_when_no_records() -> None:
    db = _make_db()
    assert db.is_completed("nonexistent") is False


def test_mark_failed_records_failure() -> None:
    db = _make_db()
    db.mark_running("task-1", "agent-1")
    db.mark_failed("task-1", "agent-1")
    assert db.is_completed("task-1") is False


def test_mark_completed_without_running() -> None:
    db = _make_db()
    db.mark_completed("task-1", "direct-complete")
    assert db.is_completed("task-1") is True


def test_run_id_scoping() -> None:
    """Operations on one run_id must not see records from another."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db_run1 = SwarmCheckpointDB(engine, run_id="run-1")
    db_run2 = SwarmCheckpointDB(engine, run_id="run-2")

    # Complete a task in run-1
    db_run1.mark_running("task-1", "agent-1")
    db_run1.mark_completed("task-1", "success")

    # run-2 should NOT see it as completed
    assert db_run2.is_completed("task-1") is False

    # run-2 marks the same task_hash running + failed
    db_run2.mark_running("task-1", "agent-2")
    db_run2.mark_failed("task-1", "agent-2")

    # run-1's record should still be completed
    assert db_run1.is_completed("task-1") is True


def test_concurrent_access(tmp_path) -> None:
    db_path = tmp_path / "concurrent.db"
    engine = create_engine(f"sqlite:///{db_path}")
    db = SwarmCheckpointDB(engine)
    errors: list[Exception] = []

    def worker(task_id: str) -> None:
        try:
            db.mark_running(task_id, "agent-1")
            db.mark_completed(task_id, "done")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(f"task-{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    for i in range(5):
        assert db.is_completed(f"task-{i}") is True
