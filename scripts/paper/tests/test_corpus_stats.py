"""Tests for corpus statistics computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.paper.corpus_stats import compute_corpus_stats


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a minimal corpus JSONL for testing."""
    records = [
        {"trajectory": "def foo():\n    return 1\n" * 50, "steps": 3},
        {"trajectory": "x = 1\n" * 200, "steps": 7},
        {"trajectory": "import os\n" * 10, "steps": 1},
    ]
    out = tmp_path / "corpus.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out


def test_stats_keys(sample_corpus: Path) -> None:
    stats = compute_corpus_stats(sample_corpus)
    assert "mean_tokens" in stats
    assert "median_tokens" in stats
    assert "p95_tokens" in stats
    assert "max_steps" in stats
    assert "pct_exceeding_4k" in stats
    assert "pct_exceeding_16k" in stats


def test_stats_ordering(sample_corpus: Path) -> None:
    stats = compute_corpus_stats(sample_corpus)
    assert stats["median_tokens"] <= stats["p95_tokens"]
    assert 0.0 <= stats["pct_exceeding_4k"] <= 100.0
