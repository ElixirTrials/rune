"""Oracle adapter cache and lookup helpers for round-2 training.

This module is intentionally CPU-safe: torch and peft are imported inside
function bodies per INFRA-05 so the module stays importable in CPU-only CI.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ORACLE_ID_PREFIX: str = "oracle_"
DIAGNOSE_BIN_KEY: str = "diagnose_pooled"


def _bin_key_for_record(record: dict[str, Any]) -> str:
    """Derive the oracle bin key for a training manifest record.

    Prefers ``metadata.phase`` + ``metadata.benchmark`` (authoritative, set
    by the corpus producer). Falls back to parsing ``task_id`` of form
    ``"<benchmark>/<problem_id>/<phase>"``. Diagnose pools across benchmarks
    to a single ``"diagnose_pooled"`` bin.

    Args:
        record: One JSONL manifest record from the corpus producer.

    Returns:
        Bin key of form ``"<phase>_<benchmark>"`` or ``"diagnose_pooled"``.

    Raises:
        ValueError: When neither metadata nor task_id provide enough info.
    """
    meta = record.get("metadata") or {}
    phase = meta.get("phase")
    benchmark = meta.get("benchmark")

    if not phase or not benchmark:
        task_id = str(record.get("task_id", ""))
        parts = task_id.split("/")
        if len(parts) >= 3:
            benchmark = benchmark or parts[0]
            phase = phase or parts[-1]

    if not phase or not benchmark:
        raise ValueError(
            f"cannot derive bin_key from record: task_id={record.get('task_id')!r}, "
            f"metadata={meta!r}"
        )

    if phase == "diagnose":
        return DIAGNOSE_BIN_KEY
    return f"{phase}_{benchmark}"
