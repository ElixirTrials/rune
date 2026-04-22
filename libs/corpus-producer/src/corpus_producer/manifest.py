"""JSONL training manifest emission for oracle bins.

Each bin's manifest is a JSONL file where every line is a training record
compatible with ``model_training.d2l_data.pairs_to_chat_messages`` and
the ``trainer.train_and_register(dataset_path=...)`` entry point.

Schema (per record):
  task_id         str   "<benchmark>/<problem_id>/<phase>"
  activation_text str   phase input (what the model sees as context)
  teacher_text    str   activation + phase output (supervised target)
  metadata        dict  provenance: phase, benchmark, problem_id, pipeline_run_id,
                        pass_at_1, rationalized, + any extra fields from artifact
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from corpus_producer.models import PhaseArtifact

logger = logging.getLogger(__name__)


def emit_bin_manifest(
    bin_key: str,
    artifacts: list[PhaseArtifact],
    out_dir: Path | str,
) -> Path:
    """Write a JSONL training manifest for one oracle bin.

    Args:
        bin_key: Oracle bin identifier (e.g. "decompose_humaneval",
            "diagnose_pooled").
        artifacts: All PhaseArtifacts for this bin.
        out_dir: Directory to write the manifest into. Created if absent.

    Returns:
        Path to the written ``.jsonl`` file.

    Raises:
        ValueError: If ``artifacts`` is empty.
    """
    if not artifacts:
        raise ValueError(f"Cannot emit manifest for empty bin {bin_key!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{bin_key}.jsonl"

    records = [art.to_manifest_record() for art in artifacts]

    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote manifest %s: %d records -> %s",
        bin_key,
        len(records),
        manifest_path,
    )
    return manifest_path


def load_bin_manifest(path: Path | str) -> list[dict[str, object]]:
    """Read a previously-emitted manifest back into memory.

    Args:
        path: Path to the ``.jsonl`` manifest file.

    Returns:
        List of record dicts, one per non-empty line.
    """
    src = Path(path)
    records: list[dict[str, object]] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
