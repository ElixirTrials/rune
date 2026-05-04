"""Compute corpus-level statistics for a mined trajectory JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def _percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = max(0, min(len(sorted_vals) - 1, int(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def _summarise(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0}
    sorted_vals = sorted(values)
    return {
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": round(statistics.fmean(values), 1),
        "median": int(statistics.median(sorted_vals)),
        "p95": _percentile(sorted_vals, 0.95),
    }


def compute_stats(input_path: Path, output_path: Path) -> dict:
    rounds_per_traj: list[int] = []
    chars_per_traj: list[int] = []
    licenses: Counter = Counter()
    n_traj = n_episodes = 0

    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            n_traj += 1
            n_eps = len(rec["episodes"])
            n_episodes += n_eps
            rounds_per_traj.append(n_eps)
            char_count = len(rec.get("task_description", ""))
            for ep in rec["episodes"]:
                char_count += len(ep.get("prior_diff", ""))
                char_count += len(ep.get("action_diff", ""))
                char_count += len(ep["feedback"].get("body", ""))
            chars_per_traj.append(char_count)
            licenses[rec["provenance"]["license"]] += 1

    stats = {
        "n_trajectories": n_traj,
        "n_episodes": n_episodes,
        "rounds_per_traj": _summarise(rounds_per_traj),
        "chars_per_traj": _summarise(chars_per_traj),
        "licenses": dict(licenses),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("data/corpus_stats.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    stats = compute_stats(args.input, args.output)
    logger.info("%s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
