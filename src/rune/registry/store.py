"""SQLite-backed registry for tracking LoRA adapter artifacts."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterRecord:
    """Metadata for a persisted LoRA adapter.

    Attributes:
        adapter_id: Unique identifier for this adapter.
        disk_path: Absolute path to the saved safetensors file.
        parent_id: ID of the adapter this was derived from, or None.
        action: Engine action name that produced this adapter.
        session_id: ID of the run session that produced this adapter.
        generation: Lineage depth (0 = root).
        created_at: Unix timestamp of creation.
    """

    adapter_id: str
    disk_path: str
    parent_id: str | None
    action: str
    session_id: str
    generation: int
    created_at: float


class AdapterRegistry:
    """Persistent registry of LoRA adapters backed by SQLite.

    Uses WAL mode for concurrent read access.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialise the registry and create the adapters table if needed.

        Args:
            conn: Open SQLite connection; ownership is transferred.
        """
        self._conn = conn
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Without a busy timeout a second concurrent writer fails immediately
        # with "database is locked" instead of waiting for the lock.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS adapters (
                adapter_id TEXT PRIMARY KEY,
                disk_path TEXT NOT NULL,
                parent_id TEXT,
                action TEXT NOT NULL,
                session_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    @classmethod
    def create(cls, db_path: str | Path) -> AdapterRegistry:
        """Open (or create) a registry at the given path.

        Args:
            db_path: File system path for the SQLite database.

        Returns:
            Initialised AdapterRegistry.
        """
        conn = sqlite3.connect(str(db_path))
        return cls(conn)

    def register(
        self,
        adapter_id: str,
        disk_path: str,
        parent_id: str | None,
        action: str,
        session_id: str,
        generation: int,
    ) -> None:
        """Insert a new adapter record with the current timestamp.

        Args:
            adapter_id: Unique identifier for the adapter.
            disk_path: Absolute path to the safetensors file on disk.
            parent_id: Parent adapter ID, or None for root adapters.
            action: Engine action name that produced this adapter.
            session_id: Run session that produced this adapter.
            generation: Lineage depth.
        """
        self._conn.execute(
            "INSERT INTO adapters VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                adapter_id,
                disk_path,
                parent_id,
                action,
                session_id,
                generation,
                time.time(),
            ),
        )
        self._conn.commit()

    def get(self, adapter_id: str) -> AdapterRecord | None:
        """Fetch a single adapter record by ID.

        Args:
            adapter_id: The adapter to look up.

        Returns:
            AdapterRecord if found, otherwise None.
        """
        row = self._conn.execute(
            "SELECT * FROM adapters WHERE adapter_id = ?", (adapter_id,)
        ).fetchone()
        return AdapterRecord(*row) if row else None

    def lineage(self, adapter_id: str) -> list[AdapterRecord]:
        """Walk the parent chain from an adapter to the root.

        Args:
            adapter_id: Starting adapter ID.

        Returns:
            Ordered list from the given adapter up to the root ancestor.
        """
        chain: list[AdapterRecord] = []
        current: str | None = adapter_id
        while current:
            record = self.get(current)
            if record is None:
                break
            chain.append(record)
            current = record.parent_id
        return chain

    def list_by_session(self, session_id: str) -> list[AdapterRecord]:
        """Return all adapters produced by a session, ordered by generation.

        Args:
            session_id: The session to filter on.

        Returns:
            List of AdapterRecord sorted by ascending generation.
        """
        rows = self._conn.execute(
            "SELECT * FROM adapters WHERE session_id = ? ORDER BY generation",
            (session_id,),
        ).fetchall()
        return [AdapterRecord(*r) for r in rows]

    def prune(self, max_age_days: int = 7) -> int:
        """Delete adapters older than max_age_days and remove their files.

        Args:
            max_age_days: Age threshold in days.

        Returns:
            Number of records deleted.
        """
        cutoff = time.time() - (max_age_days * 86400)
        rows = self._conn.execute(
            "SELECT disk_path FROM adapters WHERE created_at < ?", (cutoff,)
        ).fetchall()
        for (disk_path,) in rows:
            # disk_path may be a save_pretrained directory, not a file; unlink()
            # raises IsADirectoryError on those, which would abort the prune
            # loop before any rows are deleted. Handle both, and never let one
            # bad path stop the DELETE.
            p = Path(disk_path)
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
            except OSError as exc:
                logger.warning("prune: could not remove %s: %s", disk_path, exc)
        cursor = self._conn.execute(
            "DELETE FROM adapters WHERE created_at < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount
