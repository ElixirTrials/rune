"""Tests for contamination filter."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.paper.contamination_filter import (
    check_exact_match,
    check_repo_level,
    filter_corpus,
)


def test_exact_match_positive() -> None:
    """Detects verbatim benchmark solution in trajectory."""
    benchmark_solutions = ["def foo():\n    return 42\n"]
    trajectory = "some context\ndef foo():\n    return 42\nmore context"
    assert check_exact_match(trajectory, benchmark_solutions) is True


def test_exact_match_negative() -> None:
    """No match when solution is absent."""
    benchmark_solutions = ["def bar():\n    return 99\n"]
    trajectory = "def foo():\n    return 42\n"
    assert check_exact_match(trajectory, benchmark_solutions) is False


def test_repo_level_exclusion() -> None:
    """Excludes trajectories from repos that contain benchmark problems."""
    excluded_repos = {"owner/benchmark-repo"}
    assert check_repo_level("owner/benchmark-repo", excluded_repos) is True
    assert check_repo_level("owner/safe-repo", excluded_repos) is False


def test_filter_corpus_counts(tmp_path: Path) -> None:
    """filter_corpus returns per-benchmark exclusion counts."""
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"trajectory": "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n", "repo": "owner/safe"},
        {"trajectory": "clean trajectory", "repo": "owner/safe"},
    ]
    with corpus.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    benchmark_solutions = {
        "humaneval": ["def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"],
    }
    result = filter_corpus(corpus, benchmark_solutions, excluded_repos=set())
    assert result["humaneval"]["exact_match_excluded"] >= 1
    assert result["total_excluded"] >= 1
    assert result["total_retained"] >= 1
