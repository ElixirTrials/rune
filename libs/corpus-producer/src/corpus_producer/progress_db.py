"""SQLite resume/checkpoint table for phase corpus production.

Schema
------
Table: phase_corpus_progress
  benchmark     TEXT  NOT NULL
  problem_id    TEXT  NOT NULL
  phase         TEXT  NOT NULL
  status        TEXT  NOT NULL   -- "pending" | "running" | "done" | "failed"
  completed_at  TEXT             -- ISO-8601 UTC, NULL until status="done"
  PRIMARY KEY (benchmark, problem_id, phase)

A separate table tracks per-bin training status:
  bin_key       TEXT  PRIMARY KEY
  status        TEXT  NOT NULL   -- "pending" | "training" | "done" | "failed"
  adapter_id    TEXT             -- set once training completes
  completed_at  TEXT
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DDL_PROGRESS = """
CREATE TABLE IF NOT EXISTS phase_corpus_progress (
    benchmark    TEXT NOT NULL,
    problem_id   TEXT NOT NULL,
    phase        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    completed_at TEXT,
    PRIMARY KEY (benchmark, problem_id, phase)
);
"""

_DDL_BIN_TRAINING = """
CREATE TABLE IF NOT EXISTS bin_training_progress (
    bin_key      TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'pending',
    adapter_id   TEXT,
    completed_at TEXT
);
"""


class ProgressDB:
    """Lightweight SQLite wrapper for phase corpus producer checkpointing.

    Args:
        db_path: Path to the SQLite database file. Created if absent.
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize ProgressDB, creating tables if the file is new."""
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_DDL_PROGRESS)
        self._conn.execute(_DDL_BIN_TRAINING)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Problem-level tracking
    # ------------------------------------------------------------------

    def mark_running(self, benchmark: str, problem_id: str, phase: str) -> None:
        """Upsert a (benchmark, problem_id, phase) row as 'running'.

        Args:
            benchmark: Benchmark identifier.
            problem_id: Problem identifier.
            phase: Pipeline phase name.
        """
        self._conn.execute(
            """
            INSERT INTO phase_corpus_progress (benchmark, problem_id, phase, status)
            VALUES (?, ?, ?, 'running')
            ON CONFLICT(benchmark, problem_id, phase) DO UPDATE SET status='running'
            """,
            (benchmark, problem_id, phase),
        )
        self._conn.commit()

    def mark_done(self, benchmark: str, problem_id: str, phase: str) -> None:
        """Mark a (benchmark, problem_id, phase) row as 'done'.

        Args:
            benchmark: Benchmark identifier.
            problem_id: Problem identifier.
            phase: Pipeline phase name.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO phase_corpus_progress
                (benchmark, problem_id, phase, status, completed_at)
            VALUES (?, ?, ?, 'done', ?)
            ON CONFLICT(benchmark, problem_id, phase) DO UPDATE
            SET status='done', completed_at=excluded.completed_at
            """,
            (benchmark, problem_id, phase, now),
        )
        self._conn.commit()

    def mark_failed(self, benchmark: str, problem_id: str, phase: str) -> None:
        """Mark a (benchmark, problem_id, phase) row as 'failed'.

        Args:
            benchmark: Benchmark identifier.
            problem_id: Problem identifier.
            phase: Pipeline phase name.
        """
        self._conn.execute(
            """
            INSERT INTO phase_corpus_progress (benchmark, problem_id, phase, status)
            VALUES (?, ?, ?, 'failed')
            ON CONFLICT(benchmark, problem_id, phase) DO UPDATE SET status='failed'
            """,
            (benchmark, problem_id, phase),
        )
        self._conn.commit()

    def is_done(self, benchmark: str, problem_id: str, phase: str) -> bool:
        """Return True if the (benchmark, problem_id, phase) row has status='done'.

        Args:
            benchmark: Benchmark identifier.
            problem_id: Problem identifier.
            phase: Pipeline phase name.

        Returns:
            True if the row exists and status is 'done'.
        """
        row = self._conn.execute(
            """
            SELECT status FROM phase_corpus_progress
            WHERE benchmark=? AND problem_id=? AND phase=?
            """,
            (benchmark, problem_id, phase),
        ).fetchone()
        return row is not None and row[0] == "done"

    # ------------------------------------------------------------------
    # Bin-level training tracking
    # ------------------------------------------------------------------

    def mark_bin_training(self, bin_key: str) -> None:
        """Mark a bin as currently being trained.

        Args:
            bin_key: Oracle bin identifier.
        """
        self._conn.execute(
            """
            INSERT INTO bin_training_progress (bin_key, status)
            VALUES (?, 'training')
            ON CONFLICT(bin_key) DO UPDATE SET status='training'
            """,
            (bin_key,),
        )
        self._conn.commit()

    def mark_bin_done(self, bin_key: str, adapter_id: str) -> None:
        """Mark a bin's training as complete with the resulting adapter_id.

        Args:
            bin_key: Oracle bin identifier.
            adapter_id: Registered adapter identifier.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO bin_training_progress
                (bin_key, status, adapter_id, completed_at)
            VALUES (?, 'done', ?, ?)
            ON CONFLICT(bin_key) DO UPDATE
            SET status='done', adapter_id=excluded.adapter_id,
                completed_at=excluded.completed_at
            """,
            (bin_key, adapter_id, now),
        )
        self._conn.commit()

    def is_bin_done(self, bin_key: str) -> bool:
        """Return True if the bin's training has status='done'.

        Args:
            bin_key: Oracle bin identifier.

        Returns:
            True if the bin row exists and status is 'done'.
        """
        row = self._conn.execute(
            "SELECT status FROM bin_training_progress WHERE bin_key=?",
            (bin_key,),
        ).fetchone()
        return row is not None and row[0] == "done"

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
