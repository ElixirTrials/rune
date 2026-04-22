"""Tests for corpus_producer.progress_db.ProgressDB."""

from __future__ import annotations

import tempfile
from pathlib import Path

from corpus_producer.progress_db import ProgressDB


def _db(tmp: str) -> ProgressDB:
    return ProgressDB(Path(tmp) / "progress.db")


def test_is_done_false_before_mark():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        assert db.is_done("humaneval", "HE/0", "decompose") is False
        db.close()


def test_mark_done_then_is_done():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_done("humaneval", "HE/0", "decompose")
        assert db.is_done("humaneval", "HE/0", "decompose") is True
        db.close()


def test_mark_running_not_done():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_running("humaneval", "HE/0", "plan")
        assert db.is_done("humaneval", "HE/0", "plan") is False
        db.close()


def test_mark_failed_not_done():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_failed("humaneval", "HE/0", "code")
        assert db.is_done("humaneval", "HE/0", "code") is False
        db.close()


def test_is_done_different_phases_independent():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        db.mark_done("humaneval", "HE/0", "decompose")
        assert db.is_done("humaneval", "HE/0", "plan") is False
        db.close()


def test_bin_done_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = _db(tmp)
        assert db.is_bin_done("decompose_humaneval") is False
        db.mark_bin_training("decompose_humaneval")
        assert db.is_bin_done("decompose_humaneval") is False
        db.mark_bin_done("decompose_humaneval", "oracle_decompose_humaneval")
        assert db.is_bin_done("decompose_humaneval") is True
        db.close()


def test_db_persists_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        db1 = _db(tmp)
        db1.mark_done("mbpp", "MBPP/1", "integrate")
        db1.close()
        db2 = _db(tmp)
        assert db2.is_done("mbpp", "MBPP/1", "integrate") is True
        db2.close()
