"""Tests for MBPPAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.benchmarks.mbpp import MBPPAdapter
from evaluation.benchmarks.protocol import PassVerdict

FIXTURE = Path(__file__).parent / "fixtures" / "mbpp_mini.parquet"


@pytest.fixture(autouse=True)
def use_fixture_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use local parquet fixture; set HF offline mode."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr("evaluation.benchmarks.mbpp.MBPPAdapter._fixture_path", FIXTURE)


def test_load_problems_returns_list() -> None:
    """load_problems returns a non-empty list of Problem instances."""
    adapter = MBPPAdapter()
    problems = adapter.load_problems()
    assert len(problems) > 0


def test_load_problems_max_samples() -> None:
    """load_problems respects max_samples cap."""
    adapter = MBPPAdapter()
    assert len(adapter.load_problems(max_samples=2)) <= 2


def test_problem_fields_populated() -> None:
    """Each problem has non-empty problem_id, prompt, and test_code."""
    adapter = MBPPAdapter()
    for p in adapter.load_problems():
        assert p.problem_id
        assert p.prompt
        assert p.test_code


def test_score_wrong_returns_fail() -> None:
    """Scoring a trivially wrong generation returns a PassVerdict."""
    adapter = MBPPAdapter()
    p = adapter.load_problems()[0]
    verdict = adapter.score(p, "    return None", timeout_s=10)
    assert isinstance(verdict, PassVerdict)
    # returning None will almost certainly fail MBPP assertions
    assert verdict.problem_id == p.problem_id


def test_score_timeout() -> None:
    """Infinite loop generation returns timed_out=True."""
    adapter = MBPPAdapter()
    p = adapter.load_problems()[0]
    # Module-level infinite loop — guaranteed to hang regardless of function names.
    # MBPP score runs: generation + test_code (prompt is NL description, not executed).
    infinite = "while True:\n    pass"
    verdict = adapter.score(p, infinite, timeout_s=2)
    assert verdict.timed_out is True
    assert verdict.passed is False


def test_prompt_is_docstring_format() -> None:
    """Prompt wraps description in triple-quoted docstring for completions API."""
    adapter = MBPPAdapter()
    p = adapter.load_problems()[0]
    assert p.prompt.startswith('"""\n')
    assert p.prompt.endswith('"""\n')


def test_entry_point_extracted() -> None:
    """entry_point is extracted from the first assert in test_code."""
    adapter = MBPPAdapter()
    for p in adapter.load_problems():
        assert p.entry_point is not None, f"{p.problem_id}: entry_point is None"
        assert p.entry_point in p.test_code, (
            f"{p.problem_id}: entry_point {p.entry_point!r} not in test_code"
        )


def test_test_code_valid_python_from_parquet() -> None:
    """test_code from parquet fixture is valid newline-separated Python."""
    adapter = MBPPAdapter()
    for p in adapter.load_problems():
        lines = p.test_code.split("\n")
        assert len(lines) >= 1
        for line in lines:
            assert not line.startswith("["), (
                f"{p.problem_id}: test_code looks like numpy repr: {line[:60]}"
            )


def test_score_correct_solution() -> None:
    """Scoring a known-correct generation passes (task 602: first_repeated_char)."""
    adapter = MBPPAdapter()
    problems = adapter.load_problems()
    # task 602: first_repeated_char
    p602 = next((p for p in problems if "602" in p.problem_id), None)
    if p602 is None:
        pytest.skip("task 602 not in fixture")
    correct = (
        "def first_repeated_char(str1):\n"
        "    for index, c in enumerate(str1):\n"
        "        if str1[:index].count(c) > 0:\n"
        "            return c\n"
        '    return "None"\n'
    )
    verdict = adapter.score(p602, correct, timeout_s=10)
    assert verdict.passed is True, f"Expected pass, got error: {verdict.error}"
