"""Tests for SWEBenchLiteAdapter.

score() must raise NotImplementedError (preflight clone/apply not yet implemented).
load_problems() must work with fixture data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.swe_bench import SWEBenchLiteAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "swe_bench_lite_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.swe_bench.SWEBenchLiteAdapter._fixture_path",
        FIXTURE,
    )


def test_load_problems_returns_list() -> None:
    """load_problems() works even though score() is not implemented."""
    adapter = SWEBenchLiteAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_benchmark_id() -> None:
    """benchmark_id is 'swe_bench_lite'."""
    assert SWEBenchLiteAdapter.benchmark_id == "swe_bench_lite"


def test_problem_has_repo_in_metadata() -> None:
    """SWE-Bench problems include repo in metadata."""
    adapter = SWEBenchLiteAdapter()
    for p in adapter.load_problems():
        assert "repo" in p.metadata


def test_score_raises_not_implemented() -> None:
    """score() must raise NotImplementedError with informative message."""
    adapter = SWEBenchLiteAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0
    with pytest.raises(
        NotImplementedError, match="preflight clone/apply not yet implemented"
    ):
        adapter.score(problems[0], "some patch", timeout_s=30)
