"""Data preparation pipeline for context distillation training.

Converts raw trajectory JSON files into a training JSONL by calling
format_for_distillation on each trajectory and persisting the resulting
records via save_jsonl.

Usage (CLI):
    uv run python -m model_training.d2l_prep traj1.json traj2.json -o train.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from model_training.d2l_data import format_for_distillation, save_jsonl

logger = logging.getLogger(__name__)

__all__ = ["prepare_training_jsonl"]


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    """Load one or more trajectory dicts from a JSON file.

    Handles both a single trajectory dict and a JSON array of trajectories.

    Args:
        path: Path to a JSON file containing one trajectory dict or a list.

    Returns:
        List of trajectory dicts (length >= 0).
    """
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw  # type: ignore[return-value]
    if isinstance(raw, dict):
        return [raw]
    logger.warning(
        "Unexpected JSON type in %s: %s — skipping", path, type(raw).__name__
    )
    return []


def prepare_training_jsonl(
    input_paths: list[Path],
    output_path: Path,
) -> int:
    """Convert trajectory JSON files to a training JSONL.

    Reads each input file, calls format_for_distillation on every trajectory,
    collects all returned records, and writes them to output_path via save_jsonl.

    Failed trajectories (outcome != 'success') are filtered by
    format_for_distillation and produce zero records — they do not raise.

    Args:
        input_paths: Trajectory JSON files, each containing a single trajectory
            dict or a JSON array of trajectory dicts.
        output_path: Destination JSONL file. Parent directories are created
            automatically. File is always written (may be empty).

    Returns:
        Number of records written.
    """
    all_records: list[dict[str, Any]] = []

    for path in input_paths:
        trajectories = _load_trajectories(path)
        logger.info("Processing %s (%d trajectories)", path, len(trajectories))
        for traj in trajectories:
            records = format_for_distillation(traj)
            all_records.extend(records)
            logger.debug(
                "  trajectory %s → %d records",
                traj.get("task_id", "<no-id>"),
                len(records),
            )

    save_jsonl(all_records, output_path)
    logger.info(
        "Wrote %d records to %s", len(all_records), output_path
    )
    return len(all_records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m model_training.d2l_prep",
        description=(
            "Convert raw trajectory JSON files to a training JSONL "
            "for context distillation."
        ),
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="One or more trajectory JSON files to process.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        dest="output_path",
        help="Output JSONL file path.",
    )

    args = parser.parse_args()
    n = prepare_training_jsonl(
        input_paths=args.input_paths,
        output_path=args.output_path,
    )
    print(f"Wrote {n} records to {args.output_path}")
