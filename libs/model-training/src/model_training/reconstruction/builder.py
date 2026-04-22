"""Reconstruction dataset orchestrator.

Given an AdapterRegistry and a description callback, produce a manifest +
task_embeddings.pt (+ optional zscore_stats.pt) in ``out_dir``. Raises
**before** any file is written when the adapter corpus is heterogeneous in
``base_model_id``, ``warm_start_adapter``, ``target_modules``, ``rank``, or
``layer_indices``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from adapter_registry.models import AdapterRecord
from adapter_registry.registry import AdapterRegistry

from model_training.reconstruction.extract import (
    extract_lora_ab_from_state_dict,
    load_adapter_as_record,
    load_adapter_state_dict,
)
from model_training.reconstruction.manifest import (
    SCHEMA_VERSION,
    ReconstructionManifest,
    ReconstructionRecord,
    validate_homogeneity,
)
from model_training.reconstruction.registry_source import iter_reconstruction_candidates
from model_training.reconstruction.stats import compute_zscore_stats, save_zscore_stats
from model_training.reconstruction.task_embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    compute_task_embeddings,
    save_task_embeddings,
)

logger = logging.getLogger(__name__)


def build_reconstruction_dataset(
    *,
    registry: AdapterRegistry,
    out_dir: Path,
    task_description_fn: Callable[[AdapterRecord], str],
    warm_start_adapter: str | None,
    base_model_id_override: str | None,
    emb_model: Any | None,
    compute_zscore: bool,
    task_type: str | None = None,
    min_fitness: float | None = None,
    sources: tuple[str, ...] | None = None,
    emb_model_name: str | None = None,
    emb_model_dim: int | None = None,
) -> Path:
    """Build a reconstruction dataset from a registry into ``out_dir``.

    Args:
        registry: Open AdapterRegistry.
        out_dir: Destination directory (created if missing).
        task_description_fn: Callable returning free-text task description
            for an AdapterRecord. The result feeds the task embedder.
        warm_start_adapter: Warm-start adapter id stored on every record;
            ``None`` when no warm-start was used.
        base_model_id_override: Override for the per-adapter
            ``base_model_name_or_path`` read from ``adapter_config.json``.
            Rune's trainer stores the warm-start adapter there, so the true
            base (e.g. ``Qwen/Qwen3.5-9B``) must be supplied explicitly.
        emb_model: Pre-loaded sentence-transformer encoder, or ``None`` to
            use the one-hot fallback.
        compute_zscore: When True, compute + persist z-score stats to
            ``zscore_stats.pt`` and record the path in the manifest.
        task_type: Passed through to ``iter_reconstruction_candidates``.
        min_fitness: Passed through to ``iter_reconstruction_candidates``.
        sources: Passed through to ``iter_reconstruction_candidates``.
        emb_model_name: HF repo id recorded in the manifest; defaults to
            ``DEFAULT_EMBEDDING_MODEL`` when an emb_model is supplied.
        emb_model_dim: Embedding dim recorded in the manifest; defaults to
            ``DEFAULT_EMBEDDING_DIM`` when an emb_model is supplied.

    Returns:
        Path to the written manifest.

    Raises:
        ValueError: On empty candidate set or heterogeneous corpus.
    """
    candidates = iter_reconstruction_candidates(
        registry, task_type=task_type, min_fitness=min_fitness, sources=sources
    )
    if not candidates:
        raise ValueError(
            "no candidates found in registry"
            " (check task_type/min_fitness/sources filters)"
        )

    logger.info("build_reconstruction_dataset: %d candidate(s)", len(candidates))

    now = datetime.now(timezone.utc).isoformat()
    records: list[ReconstructionRecord] = []
    tensors_per_record: list[dict[str, dict[str, Any]]] = []
    descriptions: dict[str, str] = {}

    for rec in candidates:
        adapter_dir = Path(rec.file_path)
        description = task_description_fn(rec)
        record_kwargs = load_adapter_as_record(
            adapter_dir,
            task_id=rec.id,
            task_description=description,
            warm_start_adapter=warm_start_adapter,
            base_model_id_override=base_model_id_override,
            created_at=rec.created_at,
            source_task_hash=rec.training_task_hash,
            fitness_score=rec.fitness_score,
        )
        records.append(ReconstructionRecord(**record_kwargs))
        descriptions[rec.id] = description

        # Re-load state_dict for tensor stats (cheap: already on disk).
        if compute_zscore:
            sd = load_adapter_state_dict(adapter_dir)
            tensors_per_record.append(
                extract_lora_ab_from_state_dict(
                    sd, target_modules=record_kwargs["target_modules"]
                )
            )

    first = records[0]
    validate_homogeneity(
        records,
        base_model_id=first.base_model_id,
        warm_start_adapter=first.warm_start_adapter,
        target_modules=first.target_modules,
        rank=first.rank,
        layer_indices=first.layer_indices,
    )

    # Only now do we touch disk.
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = compute_task_embeddings(descriptions, model=emb_model)
    save_task_embeddings(embeddings, out_dir / "task_embeddings.pt")

    stats_path: Path | None = None
    if compute_zscore:
        stats = compute_zscore_stats(tensors_per_record)
        stats_path = out_dir / "zscore_stats.pt"
        save_zscore_stats(stats, stats_path)

    # Determine declared emb dim.
    if emb_model is None:
        declared_model_name = None
        declared_dim = len(descriptions)
    else:
        declared_model_name = emb_model_name or DEFAULT_EMBEDDING_MODEL
        declared_dim = emb_model_dim or DEFAULT_EMBEDDING_DIM

    manifest = ReconstructionManifest(
        schema_version=SCHEMA_VERSION,
        base_model_id=first.base_model_id,
        warm_start_adapter=first.warm_start_adapter,
        target_modules=first.target_modules,
        rank=first.rank,
        layer_indices=first.layer_indices,
        task_embedding_model=declared_model_name,
        task_embedding_dim=declared_dim,
        records=records,
        zscore_stats_path=str(stats_path.resolve()) if stats_path else None,
        created_at=now,
    )
    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)
    logger.info("wrote manifest: %s", manifest_path)
    return manifest_path


__all__ = ["build_reconstruction_dataset"]
