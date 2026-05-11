#!/usr/bin/env python3
# Ingest microsoft/codereview-data into D2L training pairs JSONL.
from __future__ import annotations

import argparse
import sys

from model_training.d2l_data import save_jsonl
from model_training.d2l_external import ingest_codereview_to_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest microsoft/codereview-data into D2L training pairs JSONL."
    )
    parser.add_argument("--output", required=True, help="Path to write output JSONL.")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Max HF rows to read."
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality score (default 0.0).",
    )
    parser.add_argument(
        "--split", default="train", help="HF dataset split (default: train)."
    )
    args = parser.parse_args()

    pairs = ingest_codereview_to_pairs(
        split=args.split,
        max_rows=args.max_rows,
        min_quality_score=args.min_quality,
    )
    save_jsonl(pairs, args.output)
    print(f"Ingested {len(pairs)} pairs → {args.output}")


if __name__ == "__main__":
    sys.exit(main())
