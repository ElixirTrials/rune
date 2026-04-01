"""GitHub trajectory mining CLI.

Mines GitHub repositories into trajectory JSON files suitable for
Doc-to-LoRA distillation. Supports single-repo mode (--repo) and
batch mode (--batch) for processing multiple repos from a config.

Usage:
    # Single repo
    uv run python scripts/mine_github.py --repo owner/repo -o out.json
    uv run python scripts/mine_github.py --repo owner/repo --quality -o prs.json

    # Batch mode (produces per-repo JSONL with training pairs)
    uv run python scripts/mine_github.py --batch instructions/mining_repos.json --output-dir data/pairs/
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
        description="Mine GitHub repositories into trajectory files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Single-repo mode ---
    parser.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        help='GitHub repository in "owner/repo" format (single-repo mode).',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Output JSON file path (single-repo mode).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize trajectories for distillation compatibility (single-repo mode).",
    )
    # --- Batch mode ---
    parser.add_argument(
        "--batch",
        type=Path,
        metavar="CONFIG",
        help="Path to JSON repos config for batch mining.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Output directory for batch mode (one JSONL per repo).",
    )
    # --- Shared options ---
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
        "--quality",
        action="store_true",
        help="Pre-filter PRs via GitHub Search API for quality.",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=1,
        metavar="N",
        help="Minimum review comments for quality filter.",
    )
    parser.add_argument(
        "--min-commits",
        type=int,
        default=2,
        metavar="N",
        help="Minimum commits for quality filter.",
    )
    parser.add_argument(
        "--exclude-labels",
        default=None,
        metavar="L1,L2,...",
        help="Comma-separated labels to exclude.",
    )
    parser.add_argument(
        "--max-diff-lines",
        type=int,
        default=500,
        metavar="N",
        help="Maximum lines per compressed diff in batch mode.",
    )

    args = parser.parse_args()

    # Validate argument combinations
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
    """Run single-repo mining (original behavior)."""
    from model_training.d2l_data import normalize_mined_trajectory
    from model_training.d2l_mining import (
        mine_issue_commit_chains,
        mine_pr_diff_chains,
        search_quality_prs,
    )

    exclude_labels: list[str] | None = None
    if args.exclude_labels:
        exclude_labels = [lbl.strip() for lbl in args.exclude_labels.split(",")]

    trajectories: list[dict] = []

    if args.mode in ("prs", "both"):
        pr_numbers: list[int] | None = None
        if args.quality:
            logger.info(
                "Searching for quality PRs in %s (min_reviews=%d, min_commits=%d)...",
                args.repo,
                args.min_reviews,
                args.min_commits,
            )
            pr_numbers = search_quality_prs(
                args.repo,
                max_results=args.max_items,
                github_token=token,
                min_review_comments=args.min_reviews,
                min_commits=args.min_commits,
                exclude_labels=exclude_labels,
            )
            logger.info("Quality filter selected %d PR(s).", len(pr_numbers))

        logger.info(
            "Mining PR diff chains from %s (max=%d)...", args.repo, args.max_items
        )
        pr_trajectories = mine_pr_diff_chains(
            args.repo,
            max_prs=args.max_items,
            github_token=token,
            pr_numbers=pr_numbers,
        )
        logger.info("Mined %d PR diff chain(s).", len(pr_trajectories))
        trajectories.extend(pr_trajectories)

    if args.mode in ("issues", "both"):
        logger.info(
            "Mining issue-commit chains from %s (max=%d)...",
            args.repo,
            args.max_items,
        )
        issue_trajectories = mine_issue_commit_chains(
            args.repo,
            max_issues=args.max_items,
            github_token=token,
        )
        logger.info("Mined %d issue-commit chain(s).", len(issue_trajectories))
        trajectories.extend(issue_trajectories)

    logger.info("Total trajectories collected: %d.", len(trajectories))

    if args.normalize:
        logger.info("Normalizing %d trajectory/trajectories...", len(trajectories))
        trajectories = [normalize_mined_trajectory(t) for t in trajectories]
        logger.info("Normalization complete.")

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(trajectories, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %d trajectory/trajectories to %s.", len(trajectories), output)


def _run_batch(
    config_path: Path,
    output_dir: Path,
    token: str,
    compress: bool = True,
    max_diff_lines: int = 500,
) -> None:
    """Run batch mining from a repos config file.

    Processes each repo in the config, extracts training pairs via
    normalize_mined_pairs, and saves per-repo JSONL files.

    Note: batch mode mines PRs only (not issues), because training pairs are
    derived from review-comment → revision cycles found in PR diff chains.
    """
    from model_training.d2l_data import normalize_mined_pairs, save_jsonl
    from model_training.d2l_mining import mine_pr_diff_chains, search_quality_prs

    config = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = config.get("defaults", {})
    repos = config.get("repos", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    total_pairs = 0

    for repo_cfg in repos:
        repo = repo_cfg["repo"]
        try:
            language = repo_cfg.get("language")
            max_prs = repo_cfg.get("max_prs", defaults.get("max_prs", 50))
            quality = repo_cfg.get("quality", defaults.get("quality", True))
            min_reviews = repo_cfg.get(
                "min_review_comments", defaults.get("min_review_comments", 1)
            )
            min_commits = repo_cfg.get("min_commits", defaults.get("min_commits", 2))

            logger.info("Mining %s (max=%d, quality=%s)...", repo, max_prs, quality)

            pr_numbers: list[int] | None = None
            if quality:
                pr_numbers = search_quality_prs(
                    repo,
                    max_results=max_prs,
                    github_token=token,
                    min_review_comments=min_reviews,
                    min_commits=min_commits,
                )
                logger.info("Quality filter: %d PRs for %s", len(pr_numbers), repo)

            trajectories = mine_pr_diff_chains(
                repo, max_prs=max_prs, github_token=token, pr_numbers=pr_numbers
            )
            logger.info("Mined %d trajectories from %s", len(trajectories), repo)

            pairs: list[dict] = []
            for traj in trajectories:
                pairs.extend(
                    normalize_mined_pairs(
                        traj,
                        compress=compress,
                        max_diff_lines=max_diff_lines,
                        language=language,
                    )
                )

            filename = repo.replace("/", "_") + ".jsonl"
            save_jsonl(pairs, output_dir / filename)
            logger.info("Saved %d pairs to %s", len(pairs), filename)
            total_pairs += len(pairs)
        except Exception:
            logger.exception("Failed to mine %s, skipping", repo)
            continue

    logger.info(
        "Batch complete: %d total pairs across %d repos", total_pairs, len(repos)
    )


def main() -> None:
    """Entry point for the mine_github CLI."""
    args = parse_args()

    token: str | None = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("No GitHub token. Use --token or set the GITHUB_TOKEN env var.")
        sys.exit(1)

    if args.batch:
        _run_batch(
            config_path=args.batch,
            output_dir=args.output_dir,
            token=token,
            max_diff_lines=args.max_diff_lines,
        )
    else:
        _run_single(args, token)


if __name__ == "__main__":
    main()
