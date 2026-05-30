"""Unit tests for MBPP benchmark task generation."""

from __future__ import annotations

from pathlib import Path

from rune.bench.mbpp import _row_to_task
from rune.bench.runner import dump_tasks, load_tasks

_ROW = {
    "task_id": 11,
    "text": "Remove first and last occurrence of a character from a string.",
    "test_list": [
        'assert remove_Occ("hello", "l") == "heo"',
        'assert remove_Occ("abcda", "a") == "bcd"',
    ],
    "test_imports": ["import math"],
}


class TestRowToTask:
    def test_basic_mapping(self) -> None:
        t = _row_to_task(_ROW)
        assert t.task_id == "mbpp/11"
        assert t.entry_point == "remove_Occ"  # regex'd from first assert
        assert "Remove first and last" in t.description
        assert ">>> assert remove_Occ" in t.description  # doctest hint
        # imports folded into test_code; asserts present
        assert t.test_code.startswith("import math")
        assert 'remove_Occ("abcda", "a")' in t.test_code

    def test_entry_point_fallback(self) -> None:
        t = _row_to_task({"task_id": 1, "text": "x", "test_list": ["print(1)"]})
        assert t.entry_point == "solution"  # no assert -> default

    def test_no_imports(self) -> None:
        t = _row_to_task({"task_id": 2, "text": "y", "test_list": ["assert f() == 1"]})
        assert t.test_code == "assert f() == 1"  # no leading setup
        assert t.entry_point == "f"


class TestDumpLoadRoundtrip:
    def test_roundtrip(self, tmp_path: Path) -> None:
        tasks = [_row_to_task(_ROW)]
        path = dump_tasks(tasks, tmp_path / "mbpp_tasks.json")
        loaded = load_tasks(path)
        assert len(loaded) == 1
        assert loaded[0] == tasks[0]  # frozen dataclass equality
