"""Unit tests for the shard-slicing helpers in phase_corpus_producer.

``--shard N/M`` splits the problem list so multiple nodes/workers can run
in parallel. The slice is round-robin to balance per-problem runtime
variance, and the union of all shards equals the full problem list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


class TestParseShard:
    """``_parse_shard('N/M')`` returns (idx, total)."""

    def test_valid_shard(self) -> None:
        import phase_corpus_producer as pcp

        assert pcp._parse_shard("0/4") == (0, 4)
        assert pcp._parse_shard("3/4") == (3, 4)

    def test_missing_slash_raises(self) -> None:
        import phase_corpus_producer as pcp

        with pytest.raises(ValueError):
            pcp._parse_shard("0-4")

    def test_non_integer_raises(self) -> None:
        import phase_corpus_producer as pcp

        with pytest.raises(ValueError):
            pcp._parse_shard("a/b")

    def test_index_out_of_range_raises(self) -> None:
        import phase_corpus_producer as pcp

        with pytest.raises(ValueError):
            pcp._parse_shard("4/4")  # idx must be < total
        with pytest.raises(ValueError):
            pcp._parse_shard("-1/4")

    def test_total_zero_raises(self) -> None:
        import phase_corpus_producer as pcp

        with pytest.raises(ValueError):
            pcp._parse_shard("0/0")


class TestApplyShard:
    """``apply_shard(problems, idx, total)`` round-robin slice."""

    def test_single_shard_identity(self) -> None:
        import phase_corpus_producer as pcp

        problems = [("p", "prompt")] * 10
        assert pcp.apply_shard(problems, 0, 1) == problems

    def test_two_shards_split_even_and_odd(self) -> None:
        import phase_corpus_producer as pcp

        problems = [(f"p{i}", f"prompt{i}") for i in range(6)]
        shard_0 = pcp.apply_shard(problems, 0, 2)
        shard_1 = pcp.apply_shard(problems, 1, 2)
        assert [p[0] for p in shard_0] == ["p0", "p2", "p4"]
        assert [p[0] for p in shard_1] == ["p1", "p3", "p5"]

    def test_shard_union_is_full_list(self) -> None:
        import phase_corpus_producer as pcp

        problems = [(f"p{i}", f"prompt{i}") for i in range(11)]
        recovered: list[tuple[str, str]] = []
        for idx in range(4):
            recovered.extend(pcp.apply_shard(problems, idx, 4))
        assert sorted(recovered) == sorted(problems)
        assert len(recovered) == len(problems)

    def test_shard_never_overlaps(self) -> None:
        import phase_corpus_producer as pcp

        problems = [(f"p{i}", f"prompt{i}") for i in range(20)]
        seen: set[str] = set()
        for idx in range(5):
            for pid, _ in pcp.apply_shard(problems, idx, 5):
                assert pid not in seen, f"duplicate problem {pid} across shards"
                seen.add(pid)
        assert len(seen) == len(problems)

    def test_shard_more_workers_than_problems(self) -> None:
        """When total > len(problems), each shard gets at most 1 problem."""
        import phase_corpus_producer as pcp

        problems = [(f"p{i}", f"prompt{i}") for i in range(3)]
        for idx in range(3):
            assert len(pcp.apply_shard(problems, idx, 10)) == 1
        for idx in range(3, 10):
            assert pcp.apply_shard(problems, idx, 10) == []

    def test_empty_problems(self) -> None:
        import phase_corpus_producer as pcp

        assert pcp.apply_shard([], 0, 4) == []
