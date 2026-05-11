#!/usr/bin/env python3
"""Abridge extreme-length records in the training JSONL dataset.

Identifies records whose estimated token count exceeds a threshold and uses
Claude Haiku to produce shortened versions that preserve all changed lines
with surrounding context.

Usage:
    uv run python scripts/abridge_extreme_records.py \
        --dataset data/mined/external_codereview.unrolled.jsonl

    uv run python scripts/abridge_extreme_records.py \
        --dataset data/mined/external_codereview.unrolled.jsonl \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.5
MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 10

SYSTEM_PROMPT = (
    "You are a code abridger. Given pre-revision and post-revision code, "
    "produce shortened versions that preserve all changes and enough context "
    "for a code review model to learn from. You MUST output exactly two "
    "sections: '## Abridged Pre' followed by the abridged pre-revision code, "
    "then '## Abridged Post' followed by the abridged post-revision code. "
    "Do not add any other commentary."
)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def build_abridge_prompt(pre_code: str, post_code: str, filename: str) -> str:
    return (
        f"## Pre-revision code ({filename})\n"
        f"{pre_code}\n\n"
        f"## Post-revision code ({filename})\n"
        f"{post_code}\n\n"
        "Instructions:\n"
        "- Keep ALL changed lines (additions, deletions, modifications) exactly as-is\n"
        "- Keep 5 lines of context above and below each change\n"
        '- Replace other unchanged sections with "// ... (N lines unchanged) ..."\n'
        "- Preserve import statements, class/function signatures even if unchanged\n"
        "- Output format:\n"
        "## Abridged Pre\n"
        "<abridged pre-revision code here>\n"
        "## Abridged Post\n"
        "<abridged post-revision code here>\n"
    )


def parse_abridge_response(response_text: str) -> tuple[str, str] | None:
    pre_match = re.search(
        r"## Abridged Pre\n(.*?)(?=\n## Abridged Post\n)",
        response_text,
        re.DOTALL,
    )
    post_match = re.search(
        r"## Abridged Post\n(.*)",
        response_text,
        re.DOTALL,
    )
    if not pre_match or not post_match:
        return None
    return pre_match.group(1).strip(), post_match.group(1).strip()


def extract_task_header(activation_text: str) -> str:
    idx = activation_text.find("## Current Code")
    if idx == -1:
        return ""
    return activation_text[:idx].rstrip()


def extract_filename_from_metadata(record: dict) -> str:
    meta = record.get("metadata", {})
    if isinstance(meta, dict):
        return meta.get("filename", "unknown_file")
    return "unknown_file"


def reconstruct_record(
    record: dict, abridged_pre: str, abridged_post: str
) -> dict:
    task_header = extract_task_header(record["activation_text"])
    new_activation = f"{task_header}\n\n## Current Code\n{abridged_pre}"
    new_teacher = f"{new_activation}\n\n## Revision\n{abridged_post}"
    return {
        **record,
        "activation_text": new_activation,
        "teacher_text": new_teacher,
        "pre_code": abridged_pre,
        "post_code": abridged_post,
    }


def abridge_record(client, record: dict) -> dict | None:  # type: ignore[type-arg]
    pre_code = record.get("pre_code", "")
    post_code = record.get("post_code", "")
    filename = extract_filename_from_metadata(record)

    user_prompt = build_abridge_prompt(pre_code, post_code, filename)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        response_text = response.content[0].text
    except Exception as exc:
        logger.warning(
            "API call failed for %s: %s", record.get("task_id", "?"), exc
        )
        return None

    parsed = parse_abridge_response(response_text)
    if parsed is None:
        logger.warning(
            "Failed to parse abridge response for %s",
            record.get("task_id", "?"),
        )
        return None

    abridged_pre, abridged_post = parsed
    return reconstruct_record(record, abridged_pre, abridged_post)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Abridge extreme-length training records using Claude Haiku."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/mined/external_codereview.unrolled.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: input stem + .abridged.jsonl)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-tokens-threshold",
        type=int,
        default=4096,
        help="Token threshold above which records get abridged",
    )
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

    output_path: Path = args.output or dataset_path.with_suffix(
        ".abridged.jsonl"
    )

    records: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), dataset_path)

    threshold = args.max_tokens_threshold
    extreme_indices: list[int] = []
    for i, rec in enumerate(records):
        activation_tokens = estimate_tokens(rec.get("activation_text", ""))
        teacher_suffix = rec.get("teacher_text", "")[
            len(rec.get("activation_text", "")):
        ]
        assistant_tokens = estimate_tokens(teacher_suffix)
        total_tokens = activation_tokens + assistant_tokens
        if total_tokens > threshold:
            extreme_indices.append(i)

    logger.info(
        "Found %d records exceeding %d token threshold",
        len(extreme_indices),
        threshold,
    )

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"DRY RUN: {len(extreme_indices)} records would be abridged")
        print(f"{'=' * 60}\n")
        for idx in extreme_indices[:20]:
            rec = records[idx]
            activation_tokens = estimate_tokens(rec.get("activation_text", ""))
            teacher_suffix = rec.get("teacher_text", "")[
                len(rec.get("activation_text", "")):
            ]
            total = activation_tokens + estimate_tokens(teacher_suffix)
            print(
                f"  [{idx:>4d}] {rec.get('task_id', '?'):<50s} "
                f"~{total:>5d} tokens"
            )
        if len(extreme_indices) > 20:
            print(f"  ... and {len(extreme_indices) - 20} more")
        print(f"\nOutput would be written to: {output_path}")
        return

    from anthropic import Anthropic  # noqa: PLC0415

    client = Anthropic()

    abridged_count = 0
    failed_count = 0

    for batch_start in range(0, len(extreme_indices), BATCH_SIZE):
        batch = extreme_indices[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(extreme_indices) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(
            "Processing batch %d/%d (%d records)",
            batch_num,
            total_batches,
            len(batch),
        )

        for idx in batch:
            rec = records[idx]
            task_id = rec.get("task_id", "?")

            result = abridge_record(client, rec)
            if result is not None:
                new_total = estimate_tokens(
                    result["activation_text"]
                ) + estimate_tokens(
                    result["teacher_text"][len(result["activation_text"]):]
                )
                old_total = estimate_tokens(
                    rec.get("activation_text", "")
                ) + estimate_tokens(
                    rec.get("teacher_text", "")[
                        len(rec.get("activation_text", "")):
                    ]
                )
                logger.info(
                    "  Abridged %s: %d -> %d tokens", task_id, old_total, new_total
                )
                records[idx] = result
                abridged_count += 1
            else:
                logger.warning("  Keeping original for %s (abridge failed)", task_id)
                failed_count += 1

            time.sleep(0.1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Done. Abridged %d records, %d failed (kept original). "
        "Output: %s (%d total records)",
        abridged_count,
        failed_count,
        output_path,
        len(records),
    )


if __name__ == "__main__":
    main()
