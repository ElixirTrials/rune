#!/usr/bin/env python3
"""Re-score quality_score in an external codereview JSONL dataset.

Reads each record, extracts feedback/before/after, recomputes quality_score
via score_external_quality, and writes the updated dataset.

Usage:
    uv run python scripts/rescore_dataset.py \
        --dataset data/mined/external_codereview.unrolled.jsonl

    uv run python scripts/rescore_dataset.py \
        --dataset data/mined/external_codereview.unrolled.jsonl \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_feedback(activation_text: str) -> str:
    marker = "## Review Feedback\n"
    idx = activation_text.find(marker)
    if idx < 0:
        return ""
    fb = activation_text[idx + len(marker) :]
    next_h = fb.find("\n## ")
    return fb[:next_h].strip() if next_h >= 0 else fb.strip()


def _extract_pre(activation_text: str) -> str:
    marker = "## Current Code\n"
    start = activation_text.find(marker)
    if start == -1:
        return ""
    body_start = start + len(marker)
    next_heading = activation_text.find("\n## ", body_start)
    if next_heading == -1:
        return activation_text[body_start:].rstrip("\n")
    return activation_text[body_start:next_heading].rstrip("\n")


def _extract_post(activation_text: str, teacher_text: str) -> str:
    if teacher_text.startswith(activation_text):
        rev = teacher_text[len(activation_text) :].lstrip("\n")
    else:
        rev = teacher_text
    first_nl = rev.find("\n")
    if first_nl != -1 and rev[:first_nl].startswith("## "):
        rev = rev[first_nl:].lstrip("\n")
    return rev


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score quality_score in a dataset JSONL.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/mined/external_codereview.unrolled.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_path: Path = args.dataset
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    output_path: Path = args.output or dataset_path

    from model_training.d2l_quality import score_external_quality  # noqa: PLC0415

    records: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), dataset_path)

    old_scores = [r.get("quality_score", 1.0) for r in records]

    for rec in records:
        at = rec.get("activation_text", "")
        tt = rec.get("teacher_text", "")
        feedback = _extract_feedback(at)
        before = _extract_pre(at)
        after = _extract_post(at, tt)

        new_score = score_external_quality(
            feedback_body=feedback,
            before_code=before,
            after_code=after,
        )
        rec["quality_score"] = new_score
        meta = rec.get("metadata")
        if isinstance(meta, dict):
            meta["quality_score"] = new_score

    new_scores = [r["quality_score"] for r in records]

    # Distribution summary
    buckets = [
        ("< 0.10", lambda s: s < 0.10),
        ("0.10 - 0.30", lambda s: 0.10 <= s < 0.30),
        ("0.30 - 0.50", lambda s: 0.30 <= s < 0.50),
        ("0.50 - 0.70", lambda s: 0.50 <= s < 0.70),
        ("0.70 - 0.90", lambda s: 0.70 <= s < 0.90),
        ("0.90 - 1.00", lambda s: 0.90 <= s <= 1.00),
    ]
    logger.info("Score distribution (old -> new):")
    for label, pred in buckets:
        old_count = sum(1 for s in old_scores if pred(s))
        new_count = sum(1 for s in new_scores if pred(s))
        logger.info("  %s: %5d -> %5d", label, old_count, new_count)
    logger.info(
        "  mean: %.3f -> %.3f",
        sum(old_scores) / len(old_scores),
        sum(new_scores) / len(new_scores),
    )

    if args.dry_run:
        logger.info("DRY RUN — no files written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d records to %s", len(records), output_path)


if __name__ == "__main__":
    main()
