"""Contamination filter: exact-match + repo-level exclusion.

Paper §4.1 commits to excluding any training trajectory that contains a
verbatim benchmark solution or originates from a repository that itself
contains benchmark problems.

Usage:
    uv run python scripts/paper/contamination_filter.py \
        --corpus data/pairs/corpus.jsonl \
        --benchmark-solutions data/benchmark_solutions.json \
        --excluded-repos data/excluded_repos.txt \
        --output evaluation_results/contamination_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_exact_match(trajectory: str, benchmark_solutions: list[str]) -> bool:
    """Check if any benchmark solution appears verbatim in the trajectory.

    Args:
        trajectory: Full trajectory text.
        benchmark_solutions: List of canonical solution strings.

    Returns:
        True if any solution is a substring of the trajectory.
    """
    for solution in benchmark_solutions:
        normalized_solution = solution.strip()
        if normalized_solution and normalized_solution in trajectory:
            return True
    return False


def check_repo_level(repo: str, excluded_repos: set[str]) -> bool:
    """Check if a repository is in the exclusion set.

    Args:
        repo: Repository identifier (e.g. "owner/name").
        excluded_repos: Set of excluded repository identifiers.

    Returns:
        True if the repo should be excluded.
    """
    return repo in excluded_repos


def filter_corpus(
    corpus_path: Path,
    benchmark_solutions: dict[str, list[str]],
    excluded_repos: set[str],
) -> dict[str, Any]:
    """Filter a corpus and return per-benchmark exclusion counts.

    Args:
        corpus_path: Path to JSONL corpus. Each line has "trajectory" and "repo".
        benchmark_solutions: {benchmark_name: [solution_strings]}.
        excluded_repos: Set of repo identifiers to exclude.

    Returns:
        Dict with per-benchmark counts and totals.
    """
    per_benchmark: dict[str, dict[str, int]] = {
        name: {"exact_match_excluded": 0, "repo_excluded": 0}
        for name in benchmark_solutions
    }
    total_excluded = 0
    total_retained = 0
    total_records = 0

    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            trajectory = record.get("trajectory", "")
            repo = record.get("repo", "")
            total_records += 1

            excluded = False

            if check_repo_level(repo, excluded_repos):
                for name in per_benchmark:
                    per_benchmark[name]["repo_excluded"] += 1
                excluded = True
            else:
                for name, solutions in benchmark_solutions.items():
                    if check_exact_match(trajectory, solutions):
                        per_benchmark[name]["exact_match_excluded"] += 1
                        excluded = True

            if excluded:
                total_excluded += 1
            else:
                total_retained += 1

    return {
        **per_benchmark,
        "total_records": total_records,
        "total_excluded": total_excluded,
        "total_retained": total_retained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Contamination filter for paper §4.1")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark-solutions", type=Path, required=True)
    parser.add_argument("--excluded-repos", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.benchmark_solutions.open() as f:
        benchmark_solutions: dict[str, list[str]] = json.load(f)

    excluded_repos: set[str] = set()
    if args.excluded_repos and args.excluded_repos.exists():
        excluded_repos = set(args.excluded_repos.read_text().strip().splitlines())

    result = filter_corpus(args.corpus, benchmark_solutions, excluded_repos)

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


if __name__ == "__main__":
    main()
