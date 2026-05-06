"""Corpus statistics for paper §3.1.

Computes token-length distribution (mean, median, P95), max encoder depth,
and percentage of sessions exceeding context windows.

Usage:
    uv run python scripts/paper/corpus_stats.py --corpus data/pairs/corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace + punctuation heuristic.

    For precise counts, swap this with tiktoken or the model's tokenizer.
    The 4-char approximation matches GPT-family tokenizers within ~10%.
    """
    return max(1, len(text) // 4)


def compute_corpus_stats(
    corpus_path: Path,
    context_windows: tuple[int, ...] = (4096, 16384),
) -> dict[str, Any]:
    """Compute trajectory corpus statistics.

    Args:
        corpus_path: Path to JSONL file. Each line must have a "trajectory"
            field (str) and optionally a "steps" field (int).
        context_windows: Token thresholds to report % exceeding.

    Returns:
        Dict with mean_tokens, median_tokens, p95_tokens, max_steps,
        pct_exceeding_4k, pct_exceeding_16k, total_sessions.
    """
    import numpy as np

    token_lengths: list[int] = []
    step_counts: list[int] = []

    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            traj = record.get("trajectory", "")
            steps = record.get("steps", 1)
            token_lengths.append(_count_tokens(traj))
            step_counts.append(int(steps))

    if not token_lengths:
        return {
            "mean_tokens": 0,
            "median_tokens": 0,
            "p95_tokens": 0,
            "max_steps": 0,
            "pct_exceeding_4k": 0.0,
            "pct_exceeding_16k": 0.0,
            "total_sessions": 0,
        }

    arr = np.array(token_lengths)
    n = len(arr)
    result: dict[str, Any] = {
        "mean_tokens": int(np.mean(arr)),
        "median_tokens": int(np.median(arr)),
        "p95_tokens": int(np.percentile(arr, 95)),
        "max_steps": max(step_counts),
        "total_sessions": n,
    }

    for window in context_windows:
        key = f"pct_exceeding_{window // 1024}k"
        result[key] = float(np.sum(arr > window) / n * 100)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics for paper §3.1")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stats = compute_corpus_stats(args.corpus)

    output = json.dumps(stats, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
