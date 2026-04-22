"""Query the AdapterRegistry for reconstruction candidates.

Pure read-only wrapper. Filters out archived adapters and any record whose
``file_path`` does not point to a readable directory. Does not touch the
adapter weights themselves — that's ``extract.load_adapter_as_record``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from adapter_registry.models import AdapterRecord
from adapter_registry.registry import AdapterRegistry

logger = logging.getLogger(__name__)


def iter_reconstruction_candidates(
    registry: AdapterRegistry,
    *,
    task_type: str | None = None,
    min_fitness: float | None = None,
    sources: tuple[str, ...] | None = None,
) -> list[AdapterRecord]:
    """Return AdapterRecords eligible for reconstruction-dataset inclusion.

    Filters:
        - ``is_archived=False`` (always enforced via ``registry.list_all``).
        - ``task_type`` matches iff provided.
        - ``fitness_score >= min_fitness`` iff provided (None-valued fitness
          scores are **excluded** when ``min_fitness`` is set).
        - ``source`` in ``sources`` iff provided.
        - ``file_path`` exists on disk (warn-and-skip otherwise).

    Args:
        registry: Open AdapterRegistry.
        task_type: Optional task_type filter.
        min_fitness: Optional minimum fitness_score.
        sources: Optional whitelist of source strings (e.g. ``("distillation",)``).

    Returns:
        List of AdapterRecord instances, insertion-order preserved.
    """
    records = registry.list_all()  # filters is_archived

    if task_type is not None:
        records = [r for r in records if r.task_type == task_type]
    if sources is not None:
        allowed = set(sources)
        records = [r for r in records if r.source in allowed]
    if min_fitness is not None:
        records = [
            r
            for r in records
            if r.fitness_score is not None and r.fitness_score >= min_fitness
        ]

    kept: list[AdapterRecord] = []
    for rec in records:
        if not Path(rec.file_path).is_dir():
            logger.warning(
                "dropping adapter %s: file_path %r is not a directory",
                rec.id,
                rec.file_path,
            )
            continue
        kept.append(rec)
    return kept


__all__ = ["iter_reconstruction_candidates"]
