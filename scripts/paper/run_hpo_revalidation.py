"""Re-run HPO after diff-loss bug fixes.

Wraps scripts/optimization/run_training_hpo.py with fixed parameters
and records results for the paper. This exists to document the exact
invocation used for reproducibility.

Usage:
    uv run python scripts/paper/run_hpo_revalidation.py \
        --dataset data/pairs/corpus.jsonl \
        --n-trials 200 \
        --output evaluation_results/hpo_revalidation.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="HPO re-validation post-bugfix")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--study-name", type=str, default="rune-hpo-postfix-v1")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/hpo_revalidation.json"))
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/optimization/run_training_hpo.py",
        "--dataset", str(args.dataset),
        "--n-trials", str(args.n_trials),
        "--study-name", args.study_name,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)

    report = {
        "study_name": args.study_name,
        "n_trials": args.n_trials,
        "dataset": str(args.dataset),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"HPO report: {args.output}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
