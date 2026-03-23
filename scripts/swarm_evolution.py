"""Evolution worker and sweep logic for swarm adapter evolution.

Provides async workers that periodically evaluate adapters, merge
top performers via TIES-Merging, and prune low-fitness adapters.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adapter_registry import AdapterRegistry

logger = logging.getLogger(__name__)

# Thresholds
MERGE_MIN_ADAPTERS = 5
MERGE_TOP_K = 3
PRUNE_FITNESS_THRESHOLD = 0.3


def evaluate_adapter(
    adapter_id: str,
    pass_rate: float,
    diversity_score: float,
    registry: "AdapterRegistry",
) -> float:
    """Evaluate an adapter and update its fitness in the registry.

    Args:
        adapter_id: ID of the adapter to evaluate.
        pass_rate: Measured pass rate.
        diversity_score: Measured diversity score.
        registry: Adapter registry for persisting results.

    Returns:
        Computed fitness score.
    """
    from evaluation.metrics import evaluate_fitness

    fitness = evaluate_fitness(adapter_id, pass_rate, diversity_score)
    registry.update_fitness(adapter_id, pass_rate=pass_rate, fitness_score=fitness)
    return fitness


def evolution_sweep(registry: "AdapterRegistry") -> dict[str, Any]:
    """Run one evolution sweep across all task types.

    For each task type with >= MERGE_MIN_ADAPTERS, TIES-merges the top-3
    adapters. Archives any adapter with fitness < PRUNE_FITNESS_THRESHOLD.

    Args:
        registry: Adapter registry to query and modify.

    Returns:
        Summary dict with merged and pruned counts per task type.
    """
    summary: dict[str, Any] = {"task_types": {}}

    for task_type in registry.get_task_types():
        all_adapters = registry.query_by_task_type(task_type)
        active = [a for a in all_adapters if not a.is_archived]

        merged_count = 0
        pruned_count = 0

        # Merge top adapters if we have enough
        if len(active) >= MERGE_MIN_ADAPTERS:
            top = registry.query_best_for_task(task_type, top_k=MERGE_TOP_K)
            top = [a for a in top if a.fitness_score is not None]
            if len(top) >= 2:
                parent_ids = [a.id for a in top]
                try:
                    _ties_merge_adapters(parent_ids, task_type, registry)
                    merged_count = 1
                except Exception:
                    logger.exception("TIES merge failed for task_type=%s", task_type)

        # Prune low-fitness adapters
        for adapter in active:
            if (
                adapter.fitness_score is not None
                and adapter.fitness_score < PRUNE_FITNESS_THRESHOLD
            ):
                registry.archive(adapter.id)
                pruned_count += 1

        summary["task_types"][task_type] = {
            "merged": merged_count,
            "pruned": pruned_count,
            "active": len(active) - pruned_count + merged_count,
        }

    return summary


def _ties_merge_adapters(
    adapter_ids: list[str],
    task_type: str,
    registry: "AdapterRegistry",
) -> str:
    """Load adapter state dicts, TIES-merge them, and register the result.

    All GPU imports are deferred (INFRA-05 pattern).

    Args:
        adapter_ids: IDs of adapters to merge.
        task_type: Task type for the merged adapter.
        registry: Registry for loading adapter paths and storing result.

    Returns:
        ID of the newly registered merged adapter.
    """
    import hashlib
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    from model_training.merging import load_adapter_state_dict, ties_merge

    adapters = [registry.retrieve_by_id(aid) for aid in adapter_ids]
    state_dicts = [load_adapter_state_dict(a.file_path) for a in adapters]
    merged_sd = ties_merge(state_dicts)

    # Save merged adapter to configurable directory
    merged_id = str(uuid.uuid4())
    adapter_base = os.environ.get("RUNE_ADAPTER_DIR")
    base_path = (
        Path(adapter_base) if adapter_base else Path.home() / ".rune" / "adapters"
    )
    output_dir = str(base_path / "merged" / merged_id)

    os.makedirs(output_dir, exist_ok=True)

    from safetensors.torch import save_file

    safetensors_path = Path(output_dir) / "adapter_model.safetensors"
    save_file(merged_sd, str(safetensors_path))

    # Compute file metadata
    file_hash = hashlib.sha256(safetensors_path.read_bytes()).hexdigest()
    file_size_bytes = safetensors_path.stat().st_size
    created_at = datetime.now(tz=timezone.utc).isoformat()

    # Register merged adapter
    from adapter_registry.models import AdapterRecord

    max_gen = max(a.generation for a in adapters)
    record = AdapterRecord(
        id=merged_id,
        version=1,
        task_type=task_type,
        base_model_id=adapters[0].base_model_id,
        rank=adapters[0].rank,
        created_at=created_at,
        file_path=str(safetensors_path),
        file_hash=file_hash,
        file_size_bytes=file_size_bytes,
        source="evolution",
        session_id="merge",
        parent_ids=json.dumps(adapter_ids),
        generation=max_gen + 1,
    )
    registry.store(record)
    return merged_id


async def evolution_worker(
    registry: "AdapterRegistry",
    interval_seconds: int = 7200,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Async worker that runs periodic evolution sweeps.

    Args:
        registry: Adapter registry to operate on.
        interval_seconds: Seconds between sweep runs.
        shutdown_event: Event to signal graceful shutdown.
    """
    while True:
        if shutdown_event and shutdown_event.is_set():
            break

        try:
            summary = evolution_sweep(registry)
            logger.info("Evolution sweep completed: %s", summary)
        except Exception:
            logger.exception("Evolution sweep failed")

        await asyncio.sleep(interval_seconds)
