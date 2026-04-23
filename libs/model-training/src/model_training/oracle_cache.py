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


def lookup_oracle_path(bin_key: str, registry: Any) -> str | None:
    """Resolve a bin key to the on-disk path of its registered oracle adapter.

    The adapter_id scheme follows :mod:`corpus_producer.trainer_bridge`:
    ``oracle_<bin_key>``. Returns ``None`` when the adapter is missing or
    archived so callers can decide whether to fall back to the base model
    or skip the record.

    Args:
        bin_key: Oracle bin key (e.g. ``"decompose_humaneval"``,
            ``"diagnose_pooled"``).
        registry: An :class:`adapter_registry.registry.AdapterRegistry`
            instance (or a mock with a compatible ``retrieve_by_id``).

    Returns:
        The adapter's ``file_path`` string, or ``None`` when missing / archived.
    """
    from adapter_registry.exceptions import AdapterNotFoundError  # noqa: PLC0415

    adapter_id = f"{ORACLE_ID_PREFIX}{bin_key}"
    try:
        record = registry.retrieve_by_id(adapter_id)
    except AdapterNotFoundError:
        logger.warning("Oracle adapter %r not found in registry", adapter_id)
        return None
    if record.is_archived:
        logger.warning("Oracle adapter %r is archived; ignoring", adapter_id)
        return None
    return str(record.file_path)


def audit_oracle_coverage(
    records: list[dict[str, Any]],
    registry: Any,
) -> tuple[float, dict[str, int]]:
    """Compute oracle-coverage ratio and per-bin record counts.

    Iterates the records, derives each bin_key, and checks whether a
    registered (non-archived) oracle exists for it. Caches lookup results
    per bin_key to avoid repeated registry queries.

    Args:
        records: List of JSONL manifest records.
        registry: AdapterRegistry instance.

    Returns:
        Tuple ``(coverage_ratio, bin_counts)`` where:
        - ``coverage_ratio`` is ``covered / len(records)`` (using the
          **original** record count as the denominator). Unroutable records
          — those raising ``ValueError`` from :func:`_bin_key_for_record` —
          are logged and skipped, but still count against the denominator.
          Returns ``0.0`` when ``records`` is empty.
        - ``bin_counts`` maps bin_key → record count for *routable* records
          only; unroutable records are excluded, so
          ``sum(bin_counts.values())`` may be less than ``len(records)``.
    """
    if not records:
        return 0.0, {}

    bin_counts: dict[str, int] = {}
    lookup_cache: dict[str, bool] = {}
    covered = 0

    for record in records:
        try:
            bin_key = _bin_key_for_record(record)
        except ValueError as exc:
            logger.warning("Skipping unroutable record: %s", exc)
            continue
        bin_counts[bin_key] = bin_counts.get(bin_key, 0) + 1

        if bin_key not in lookup_cache:
            lookup_cache[bin_key] = lookup_oracle_path(bin_key, registry) is not None
        if lookup_cache[bin_key]:
            covered += 1

    return covered / len(records), bin_counts
