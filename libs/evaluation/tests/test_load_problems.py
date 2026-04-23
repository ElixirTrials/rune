"""Tests for module-level ``evaluation.benchmarks.load_problems``.

This function is the APPS-stratification parity surface between Plan A
(benchmark harness) and Plan C (phase corpus producer). Plan C relies on
``load_problems("apps", max_samples=N)`` to return a stratified sample
by difficulty identical to what ``run_benchmark`` would see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APPS_FIXTURE = Path(__file__).parent / "fixtures" / "apps_mini.parquet"
HUMANEVAL_FIXTURE = Path(__file__).parent / "fixtures" / "humaneval_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixtures_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixtures; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        "evaluation.benchmarks.apps.APPSAdapter._fixture_path", APPS_FIXTURE
    )
    monkeypatch.setattr(
        "evaluation.benchmarks.humaneval.HumanEvalAdapter._fixture_path",
        HUMANEVAL_FIXTURE,
    )


def test_load_problems_apps_delegates_to_adapter() -> None:
    """load_problems('apps', max_samples=N) uses the APPSAdapter stratified sampler."""
    from evaluation.benchmarks import load_problems
    from evaluation.benchmarks.apps import APPSAdapter

    expected = APPSAdapter().load_problems(max_samples=3, seed=42)
    got = load_problems("apps", max_samples=3, seed=42)
    assert [p.problem_id for p in got] == [p.problem_id for p in expected]


def test_load_problems_apps_stratified_seed_determinism() -> None:
    """Same seed produces same ordering across load_problems calls."""
    from evaluation.benchmarks import load_problems

    a = load_problems("apps", max_samples=3, seed=42)
    b = load_problems("apps", max_samples=3, seed=42)
    assert [p.problem_id for p in a] == [p.problem_id for p in b]


def test_load_problems_humaneval_max_samples() -> None:
    """load_problems with max_samples returns at most N problems."""
    from evaluation.benchmarks import load_problems

    got = load_problems("humaneval", max_samples=2)
    assert len(got) <= 2


def test_load_problems_problem_ids_filter() -> None:
    """problem_ids restricts the returned list to matching ids."""
    from evaluation.benchmarks import load_problems

    all_problems = load_problems("humaneval")
    target = [all_problems[0].problem_id]
    got = load_problems("humaneval", problem_ids=target)
    assert len(got) == 1
    assert got[0].problem_id == target[0]


def test_load_problems_unknown_benchmark_raises() -> None:
    """Unknown benchmark id raises ValueError."""
    from evaluation.benchmarks import load_problems

    with pytest.raises(ValueError, match="unknown"):
        load_problems("unknown")
