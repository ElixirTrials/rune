"""GitHub trajectory mining CLI.

Mines GitHub repositories into trajectory JSON files suitable for
Doc-to-LoRA distillation. Extracts PR diff chains and/or issue-commit
chains and optionally normalizes them for the distillation pipeline.

Usage:
    uv run python scripts/mine_github.py --repo owner/repo -o out.json
    uv run python scripts/mine_github.py --repo owner/repo --mode prs --max 50 -o prs.json
    uv run python scripts/mine_github.py --repo owner/repo --mode issues --normalize -o issues.json
    uv run python scripts/mine_github.py --repo owner/repo --token ghp_xxx -o both.json
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mine a GitHub repository into trajectory JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        required=True,
        metavar="OWNER/REPO",
        help='GitHub repository in "owner/repo" format.',
    )
    parser.add_argument(
        "--mode",
        choices=["prs", "issues", "both"],
        default="both",
        help="Mining mode: pull requests, issues, or both.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=100,
        dest="max_items",
        metavar="N",
        help="Maximum PRs/issues to process.",
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="GitHub personal access token (falls back to GITHUB_TOKEN env var).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        metavar="FILE",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize trajectories for distillation compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the mine_github CLI."""
    args = parse_args()

    # Resolve GitHub token
    token: str | None = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error(
            "No GitHub token provided. Use --token or set the GITHUB_TOKEN env var."
        )
        sys.exit(1)

    from model_training.d2l_data import normalize_mined_trajectory
    from model_training.d2l_mining import mine_issue_commit_chains, mine_pr_diff_chains

    trajectories: list[dict] = []

    # Mine PR diff chains
    if args.mode in ("prs", "both"):
        logger.info(
            "Mining PR diff chains from %s (max=%d)...", args.repo, args.max_items
        )
        pr_trajectories = mine_pr_diff_chains(
            args.repo,
            max_prs=args.max_items,
            github_token=token,
        )
        logger.info("Mined %d PR diff chain(s).", len(pr_trajectories))
        trajectories.extend(pr_trajectories)

    # Mine issue-commit chains
    if args.mode in ("issues", "both"):
        logger.info(
            "Mining issue-commit chains from %s (max=%d)...", args.repo, args.max_items
        )
        issue_trajectories = mine_issue_commit_chains(
            args.repo,
            max_issues=args.max_items,
            github_token=token,
        )
        logger.info("Mined %d issue-commit chain(s).", len(issue_trajectories))
        trajectories.extend(issue_trajectories)

    logger.info("Total trajectories collected: %d.", len(trajectories))

    # Normalize if requested
    if args.normalize:
        logger.info("Normalizing %d trajectory/trajectories...", len(trajectories))
        trajectories = [normalize_mined_trajectory(t) for t in trajectories]
        logger.info("Normalization complete.")

    # Write output
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(trajectories, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %d trajectory/trajectories to %s.", len(trajectories), output)


if __name__ == "__main__":
    main()
