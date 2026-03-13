"""Shared SQLAlchemy/SQLite utilities for all Rune services.

Provides:
- set_wal_mode: register WAL-mode PRAGMA on a SQLAlchemy Engine (DRY for all DBs)
- create_service_engine: build an Engine from DATABASE_URL env or a default URL
"""

from __future__ import annotations

import os

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import create_engine


def set_wal_mode(engine: Engine) -> None:
    """Register a SQLAlchemy connect event that enables WAL journal mode.

    WAL mode allows concurrent readers while a writer is active, which is
    important for swarm workloads that read checkpoints while writing.
    Safe to call multiple times on different engines; each call registers
    a separate listener on its own engine instance.

    Args:
        engine: SQLAlchemy Engine to configure.
    """

    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn: object, _record: object) -> None:  # type: ignore[type-arg]
        assert hasattr(dbapi_conn, "execute")
        dbapi_conn.execute("PRAGMA journal_mode=WAL")  # type: ignore[union-attr]


def create_service_engine(default_url: str) -> Engine:
    """Create a SQLAlchemy Engine from DATABASE_URL env var or a default URL.

    All three Rune services (api-service, training-svc, evolution-svc) use the
    same pattern: read DATABASE_URL from the environment, falling back to a
    service-specific SQLite file. This helper centralises that pattern.

    Args:
        default_url: Fallback DB URL when DATABASE_URL env var is not set.
            Typically a SQLite path, e.g. ``"sqlite:///training_svc.db"``.

    Returns:
        A configured SQLAlchemy Engine.

    Example:
        >>> engine = create_service_engine("sqlite:///training_svc.db")
    """
    db_url = os.getenv("DATABASE_URL", default_url)
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args, echo=False)
