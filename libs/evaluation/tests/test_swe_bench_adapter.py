"""Tests for SWEBenchLiteAdapter.

- load_problems() works with fixture data and surfaces scoring metadata.
- score() defaults to NotImplementedError (CI contract) unless the
  RUNE_SWE_BENCH_SCORE env var is set.
- When enabled, score() performs a clone/apply/pytest pipeline; the pipeline
  helpers are monkeypatched here so the tests stay hermetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from evaluation.benchmarks import swe_bench as swe_bench_mod
from evaluation.benchmarks.swe_bench import SWEBenchLiteAdapter, _CmdResult

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
    """load_problems() works even though scoring is env-gated."""
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


def test_problem_includes_scoring_metadata() -> None:
    """Scoring-related fields are surfaced in metadata."""
    adapter = SWEBenchLiteAdapter()
    p = adapter.load_problems()[0]
    for key in ("test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "base_commit"):
        assert key in p.metadata, key


def test_score_raises_not_implemented_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without RUNE_SWE_BENCH_SCORE, score() preserves the CI contract."""
    monkeypatch.delenv("RUNE_SWE_BENCH_SCORE", raising=False)
    adapter = SWEBenchLiteAdapter()
    problems = adapter.load_problems()
    with pytest.raises(
        NotImplementedError, match="preflight clone/apply not yet implemented"
    ):
        adapter.score(problems[0], "some patch", timeout_s=30)


def test_score_returns_pass_when_pipeline_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked clone + apply + pytest all return 0 -> PASS verdict."""
    monkeypatch.setenv("RUNE_SWE_BENCH_SCORE", "1")
    ok = _CmdResult(returncode=0, stdout="", stderr="", timed_out=False)
    monkeypatch.setattr(
        swe_bench_mod, "_run_git_clone", lambda *a, **kw: ok
    )
    monkeypatch.setattr(swe_bench_mod, "_apply_patch", lambda *a, **kw: ok)
    monkeypatch.setattr(
        swe_bench_mod, "_run_pip_install_editable", lambda *a, **kw: ok
    )
    monkeypatch.setattr(swe_bench_mod, "_run_pytest", lambda *a, **kw: ok)

    adapter = SWEBenchLiteAdapter()
    problem = adapter.load_problems()[0]
    verdict = adapter.score(problem, "dummy patch", timeout_s=30)
    assert verdict.passed is True
    assert verdict.problem_id == problem.problem_id


def test_score_fails_when_patch_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation that does not apply cleanly returns FAIL, not a raise."""
    monkeypatch.setenv("RUNE_SWE_BENCH_SCORE", "1")
    ok = _CmdResult(returncode=0, stdout="", stderr="", timed_out=False)
    reject = _CmdResult(
        returncode=1, stdout="", stderr="patch does not apply", timed_out=False
    )
    call_count = {"apply": 0}

    def fake_apply(*_args: Any, **_kw: Any) -> _CmdResult:
        call_count["apply"] += 1
        # First call is test_patch (should succeed), second is generation
        return ok if call_count["apply"] == 1 else reject

    monkeypatch.setattr(swe_bench_mod, "_run_git_clone", lambda *a, **kw: ok)
    monkeypatch.setattr(swe_bench_mod, "_apply_patch", fake_apply)

    adapter = SWEBenchLiteAdapter()
    problem = adapter.load_problems()[0]
    verdict = adapter.score(problem, "broken patch", timeout_s=30)
    assert verdict.passed is False
    assert "generation did not apply" in (verdict.error or "")


def test_score_returns_timed_out_on_pytest_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pytest timeout surfaces as timed_out=True, passed=False."""
    monkeypatch.setenv("RUNE_SWE_BENCH_SCORE", "1")
    ok = _CmdResult(returncode=0, stdout="", stderr="", timed_out=False)
    timeout = _CmdResult(
        returncode=-1, stdout="", stderr="", timed_out=True
    )
    monkeypatch.setattr(swe_bench_mod, "_run_git_clone", lambda *a, **kw: ok)
    monkeypatch.setattr(swe_bench_mod, "_apply_patch", lambda *a, **kw: ok)
    monkeypatch.setattr(
        swe_bench_mod, "_run_pip_install_editable", lambda *a, **kw: ok
    )
    monkeypatch.setattr(swe_bench_mod, "_run_pytest", lambda *a, **kw: timeout)

    adapter = SWEBenchLiteAdapter()
    problem = adapter.load_problems()[0]
    verdict = adapter.score(problem, "dummy patch", timeout_s=30)
    assert verdict.passed is False
    assert verdict.timed_out is True


def test_score_fails_fast_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing repo/base_commit is reported as a FAIL verdict, not a raise."""
    monkeypatch.setenv("RUNE_SWE_BENCH_SCORE", "1")
    adapter = SWEBenchLiteAdapter()
    bad_problem = adapter.load_problems()[0]
    # Strip scoring metadata
    bad_problem.metadata.clear()
    verdict = adapter.score(bad_problem, "patch", timeout_s=5)
    assert verdict.passed is False
    assert "missing repo" in (verdict.error or "")
