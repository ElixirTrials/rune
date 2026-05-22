from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterRecord:
    adapter_id: str
    disk_path: str
    parent_id: str | None
    action: str
    session_id: str
    generation: int
    created_at: float


class AdapterRegistry:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA journal_mode=WAL")
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
        row = self._conn.execute(
            "SELECT * FROM adapters WHERE adapter_id = ?", (adapter_id,)
        ).fetchone()
        return AdapterRecord(*row) if row else None

    def lineage(self, adapter_id: str) -> list[AdapterRecord]:
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
        rows = self._conn.execute(
            "SELECT * FROM adapters WHERE session_id = ? ORDER BY generation",
            (session_id,),
        ).fetchall()
        return [AdapterRecord(*r) for r in rows]

    def prune(self, max_age_days: int = 7) -> int:
        cutoff = time.time() - (max_age_days * 86400)
        rows = self._conn.execute(
            "SELECT disk_path FROM adapters WHERE created_at < ?", (cutoff,)
        ).fetchall()
        for (disk_path,) in rows:
            Path(disk_path).unlink(missing_ok=True)
        cursor = self._conn.execute(
            "DELETE FROM adapters WHERE created_at < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def _backdate(self, adapter_id: str, days: int) -> None:
        self._conn.execute(
            "UPDATE adapters SET created_at = ? WHERE adapter_id = ?",
            (time.time() - days * 86400, adapter_id),
        )
        self._conn.commit()
