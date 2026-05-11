"""GitHub trajectory mining CLI.

Mines GitHub repositories into trajectory JSONL files suitable for
hypernetwork training. Each line is one PR's :class:`Trajectory` record.
The companion ``unrolled.jsonl`` (batch mode) holds per-step SFT pairs
for the Direct-PEFT-QLoRA Gate-1 baseline.

Usage:
    # Single repo
    uv run python scripts/mine_github.py --repo owner/repo -o trajectories.jsonl
    uv run python scripts/mine_github.py --repo owner/repo --quality -o trajectories.jsonl

    # Batch mode
    uv run python scripts/mine_github.py --batch instructions/mining_repos.json --output-dir data/mined/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine GitHub repositories into trajectory JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", metavar="OWNER/REPO")
    parser.add_argument("-o", "--output", type=Path, metavar="FILE")
    parser.add_argument("--batch", type=Path, metavar="CONFIG")
    parser.add_argument("--output-dir", type=Path, metavar="DIR")
    parser.add_argument("--max", type=int, default=100, dest="max_items")
    parser.add_argument("--token", default=None)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--min-reviews", type=int, default=1)
    parser.add_argument("--min-commits", type=int, default=2)
    parser.add_argument("--exclude-labels", default=None)

    args = parser.parse_args()

    if args.batch:
        if not args.output_dir:
            parser.error("--output-dir is required with --batch")
    elif args.repo:
        if not args.output:
            parser.error("-o/--output is required with --repo")
    else:
        parser.error("Either --repo or --batch is required")

    return args


def _run_single(args: argparse.Namespace, token: str) -> None:
    from model_training.d2l_mining import mine_pr_trajectories, search_quality_prs_v2

    pr_numbers = None
    if args.quality:
        logger.info("Searching for quality PRs in %s ...", args.repo)
        pr_numbers = search_quality_prs_v2(
            args.repo,
            max_results=args.max_items,
            github_token=token,
        )
        logger.info("Quality filter selected %d PR(s)", len(pr_numbers))

    trajectories = mine_pr_trajectories(
        args.repo,
        pr_numbers=pr_numbers,
        max_prs=args.max_items,
        github_token=token,
    )
    logger.info("Mined %d trajectories from %s", len(trajectories), args.repo)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            fh.write(traj.model_dump_json() + "\n")


def _run_batch(config_path: Path, output_dir: Path, token: str) -> None:
    from model_training.d2l_data import save_jsonl, unroll_trajectory_to_pairs
    from model_training.d2l_mining import mine_pr_trajectories, search_quality_prs_v2

    config = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = config.get("defaults", {})
    repos = config.get("repos", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    total_traj = total_pairs = 0

    for repo_cfg in repos:
        repo = repo_cfg["repo"]
        try:
            max_prs = repo_cfg.get("max_prs", defaults.get("max_prs", 50))
            quality = repo_cfg.get("quality", defaults.get("quality", True))
            logger.info("Mining %s (max=%d, quality=%s)...", repo, max_prs, quality)

            pr_numbers = None
            if quality:
                pr_numbers = search_quality_prs_v2(
                    repo, max_results=max_prs, github_token=token
                )

            trajectories = mine_pr_trajectories(
                repo, pr_numbers=pr_numbers, max_prs=max_prs, github_token=token
            )
            logger.info("Mined %d trajectories from %s", len(trajectories), repo)

            traj_path = output_dir / f"{repo.replace('/', '_')}.trajectories.jsonl"
            with traj_path.open("w", encoding="utf-8") as fh:
                for traj in trajectories:
                    fh.write(traj.model_dump_json() + "\n")

            pairs: list[dict] = []
            for traj in trajectories:
                pairs.extend(unroll_trajectory_to_pairs(traj))
            pairs_path = output_dir / f"{repo.replace('/', '_')}.unrolled.jsonl"
            save_jsonl(pairs, pairs_path)

            total_traj += len(trajectories)
            total_pairs += len(pairs)
        except Exception:
            logger.exception("Failed to mine %s, skipping", repo)
            continue

    logger.info(
        "Batch complete: %d trajectories / %d pairs across %d repos",
        total_traj,
        total_pairs,
        len(repos),
    )


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("No GitHub token. Use --token or set GITHUB_TOKEN.")
        sys.exit(1)
    if args.batch:
        _run_batch(args.batch, args.output_dir, token)
    else:
        _run_single(args, token)


if __name__ == "__main__":
    main()
