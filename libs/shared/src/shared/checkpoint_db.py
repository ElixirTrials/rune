"""SQLite-backed checkpoint database for swarm task tracking.

Provides SwarmCheckpointDB for recording task execution state across
swarm agents, enabling crash recovery and deduplication.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, select


class CheckpointRecord(SQLModel, table=True):
    """Persistent checkpoint record for a swarm task.

    Attributes:
        id: Auto-incremented primary key.
        run_id: Swarm run identifier.
        task_hash: Hash of the task being tracked.
        agent_id: Agent executing the task.
        status: Current status (pending, running, completed, failed).
        outcome: Result description when completed.
        started_at: ISO 8601 timestamp when execution began.
        completed_at: ISO 8601 timestamp when execution finished.
    """

    __tablename__ = "swarm_checkpoints"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str
    task_hash: str = Field(index=True)
    agent_id: str
    status: str = "pending"
    outcome: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class SwarmCheckpointDB:
    """SQLite-backed checkpoint database for swarm task tracking.

    Tracks task execution state (pending → running → completed/failed)
    to enable crash recovery and prevent duplicate work.

    Example:
        >>> from sqlmodel import create_engine
        >>> engine = create_engine("sqlite:///checkpoints.db")
        >>> db = SwarmCheckpointDB(engine)
        >>> db.mark_running("task-hash-1", "agent-01")
        >>> db.mark_completed("task-hash-1", "success: pass_rate=0.95")
    """

    def __init__(self, engine: Engine) -> None:
        """Initialize checkpoint database and create tables.

        Args:
            engine: SQLAlchemy Engine connected to the target SQLite database.
        """
        self._engine = engine

        @event.listens_for(engine, "connect")
        def _set_wal(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        SQLModel.metadata.create_all(engine)

    def mark_running(self, task_hash: str, agent_id: str) -> None:
        """Record that an agent has started working on a task.

        Args:
            task_hash: Hash of the task being started.
            agent_id: Identifier of the agent executing the task.
        """
        now = datetime.now(timezone.utc).isoformat()
        record = CheckpointRecord(
            run_id="default",
            task_hash=task_hash,
            agent_id=agent_id,
            status="running",
            started_at=now,
        )
        with Session(self._engine, expire_on_commit=False) as session:
            session.add(record)
            session.commit()

    def mark_completed(self, task_hash: str, outcome: str) -> None:
        """Record that a task has been completed.

        Args:
            task_hash: Hash of the completed task.
            outcome: Description of the result.
        """
        now = datetime.now(timezone.utc).isoformat()
        with Session(self._engine, expire_on_commit=False) as session:
            stmt = (
                select(CheckpointRecord)
                .where(
                    CheckpointRecord.task_hash == task_hash,
                    CheckpointRecord.status == "running",
                )
                .limit(1)
            )
            record = session.exec(stmt).first()
            if record is not None:
                record.status = "completed"
                record.outcome = outcome
                record.completed_at = now
                session.add(record)
                session.commit()
            else:
                # No running record found; create a completed one directly
                new_record = CheckpointRecord(
                    run_id="default",
                    task_hash=task_hash,
                    agent_id="unknown",
                    status="completed",
                    outcome=outcome,
                    completed_at=now,
                )
                session.add(new_record)
                session.commit()

    def mark_failed(self, task_hash: str, agent_id: str) -> None:
        """Record that a task execution has failed.

        Args:
            task_hash: Hash of the failed task.
            agent_id: Identifier of the agent that failed.
        """
        now = datetime.now(timezone.utc).isoformat()
        with Session(self._engine, expire_on_commit=False) as session:
            stmt = (
                select(CheckpointRecord)
                .where(
                    CheckpointRecord.task_hash == task_hash,
                    CheckpointRecord.agent_id == agent_id,
                    CheckpointRecord.status == "running",
                )
                .limit(1)
            )
            record = session.exec(stmt).first()
            if record is not None:
                record.status = "failed"
                record.completed_at = now
                session.add(record)
                session.commit()
            else:
                new_record = CheckpointRecord(
                    run_id="default",
                    task_hash=task_hash,
                    agent_id=agent_id,
                    status="failed",
                    completed_at=now,
                )
                session.add(new_record)
                session.commit()

    def is_completed(self, task_hash: str) -> bool:
        """Check if a task has been completed.

        Args:
            task_hash: Hash of the task to check.

        Returns:
            True if the task has a "completed" checkpoint record.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            stmt = (
                select(CheckpointRecord)
                .where(
                    CheckpointRecord.task_hash == task_hash,
                    CheckpointRecord.status == "completed",
                )
                .limit(1)
            )
            return session.exec(stmt).first() is not None
