"""Tests for APPSAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.apps import APPSAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "apps_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr("evaluation.benchmarks.apps.APPSAdapter._fixture_path", FIXTURE)


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list of Problem instances."""
    adapter = APPSAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_load_problems_max_samples_cap() -> None:
    """max_samples is respected."""
    adapter = APPSAdapter()
    problems = adapter.load_problems(max_samples=2)
    assert len(problems) <= 2


def test_stratified_sample_seed_deterministic() -> None:
    """Same seed produces same problem ordering."""
    adapter = APPSAdapter()
    a = adapter.load_problems(max_samples=3, seed=42)
    b = adapter.load_problems(max_samples=3, seed=42)
    assert [p.problem_id for p in a] == [p.problem_id for p in b]


def test_problem_has_difficulty_in_metadata() -> None:
    """Problems from APPS include difficulty in metadata."""
    adapter = APPSAdapter()
    for p in adapter.load_problems():
        assert "difficulty" in p.metadata


def test_score_wrong_returns_verdict() -> None:
    """score() returns a PassVerdict for any generation."""
    adapter = APPSAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    print('wrong')", timeout_s=10)
    assert verdict.problem_id == p.problem_id
