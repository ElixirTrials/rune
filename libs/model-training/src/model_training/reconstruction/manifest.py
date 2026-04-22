"""Dataclasses and JSON serialization for the reconstruction training manifest.

This module is CPU-only: no torch / safetensors imports. Downstream consumers
(extract.py, builder.py) depend on these types but this file has no external
deps beyond the standard library.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReconstructionRecord:
    """One oracle adapter's index entry in the reconstruction dataset.

    Attributes:
        task_id: Stable identifier — reuses ``AdapterRecord.id``.
        adapter_path: Absolute path to the adapter directory (must contain
            ``adapter_model.safetensors`` and ``adapter_config.json``).
        task_description: Free-text description used to compute the task
            embedding. Source is caller-supplied (see builder's
            ``task_description_fn``).
        base_model_id: HF repo id of the base the adapter was trained on.
            Must match across all records in a manifest.
        warm_start_adapter: HF repo id or path of the warm-start adapter, or
            ``None`` if trained from scratch. Critical for DeltaCoder-relative
            delta semantics — see plan §"DeltaCoder-Relative Semantics".
        rank: LoRA rank. Must match across all records.
        target_modules: Names of modules the LoRA adapter targets. Must match
            across all records in a manifest.
        layer_indices: Transformer layer indices present in the adapter
            (derived from safetensor key parse). Must match across all records.
        created_at: ISO-8601 UTC timestamp (from ``AdapterRecord.created_at``).
        source_task_hash: Optional dedup key (``AdapterRecord.training_task_hash``).
        fitness_score: Optional evaluation fitness (``AdapterRecord.fitness_score``).
    """

    task_id: str
    adapter_path: str
    task_description: str
    base_model_id: str
    warm_start_adapter: str | None
    rank: int
    target_modules: tuple[str, ...]
    layer_indices: tuple[int, ...]
    created_at: str
    source_task_hash: str | None = None
    fitness_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for JSON."""
        return {
            "task_id": self.task_id,
            "adapter_path": self.adapter_path,
            "task_description": self.task_description,
            "base_model_id": self.base_model_id,
            "warm_start_adapter": self.warm_start_adapter,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "layer_indices": list(self.layer_indices),
            "created_at": self.created_at,
            "source_task_hash": self.source_task_hash,
            "fitness_score": self.fitness_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReconstructionRecord:
        """Inverse of ``to_dict``."""
        return cls(
            task_id=payload["task_id"],
            adapter_path=payload["adapter_path"],
            task_description=payload["task_description"],
            base_model_id=payload["base_model_id"],
            warm_start_adapter=payload["warm_start_adapter"],
            rank=int(payload["rank"]),
            target_modules=tuple(payload["target_modules"]),
            layer_indices=tuple(int(i) for i in payload["layer_indices"]),
            created_at=payload["created_at"],
            source_task_hash=payload.get("source_task_hash"),
            fitness_score=payload.get("fitness_score"),
        )


@dataclass(frozen=True)
class ReconstructionManifest:
    """Top-level index for a reconstruction training dataset.

    Attributes:
        schema_version: Bump on any backward-incompatible change.
        base_model_id: Homogeneous across all records.
        warm_start_adapter: Homogeneous across all records (or ``None`` if
            no warm-start was used for any oracle).
        target_modules: Homogeneous across all records.
        rank: Homogeneous across all records.
        layer_indices: Homogeneous across all records.
        task_embedding_model: HF repo id of the sentence-transformer used to
            compute embeddings, or ``None`` for the one-hot fallback.
        task_embedding_dim: Dimensionality of the task embedding vectors.
        records: One record per oracle adapter.
        zscore_stats_path: Path to optional z-score stats file (``.pt``), or
            ``None`` if stats were not computed.
        created_at: ISO-8601 UTC timestamp of manifest creation.
    """

    schema_version: int
    base_model_id: str
    warm_start_adapter: str | None
    target_modules: tuple[str, ...]
    rank: int
    layer_indices: tuple[int, ...]
    task_embedding_model: str | None
    task_embedding_dim: int
    records: list[ReconstructionRecord] = field(default_factory=list)
    zscore_stats_path: str | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for JSON."""
        return {
            "schema_version": self.schema_version,
            "base_model_id": self.base_model_id,
            "warm_start_adapter": self.warm_start_adapter,
            "target_modules": list(self.target_modules),
            "rank": self.rank,
            "layer_indices": list(self.layer_indices),
            "task_embedding_model": self.task_embedding_model,
            "task_embedding_dim": self.task_embedding_dim,
            "records": [r.to_dict() for r in self.records],
            "zscore_stats_path": self.zscore_stats_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReconstructionManifest:
        """Inverse of ``to_dict``."""
        return cls(
            schema_version=int(payload["schema_version"]),
            base_model_id=payload["base_model_id"],
            warm_start_adapter=payload["warm_start_adapter"],
            target_modules=tuple(payload["target_modules"]),
            rank=int(payload["rank"]),
            layer_indices=tuple(int(i) for i in payload["layer_indices"]),
            task_embedding_model=payload["task_embedding_model"],
            task_embedding_dim=int(payload["task_embedding_dim"]),
            records=[ReconstructionRecord.from_dict(r) for r in payload["records"]],
            zscore_stats_path=payload.get("zscore_stats_path"),
            created_at=payload.get("created_at", ""),
        )

    def save(self, path: Path) -> None:
        """Write manifest as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ReconstructionManifest:
        """Read manifest from a JSON file."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def validate_homogeneity(
    records: list[ReconstructionRecord],
    *,
    base_model_id: str,
    warm_start_adapter: str | None,
    target_modules: tuple[str, ...],
    rank: int,
    layer_indices: tuple[int, ...],
) -> None:
    """Assert every record matches the given homogeneous fields.

    Raises:
        ValueError: With a message naming the first mismatched field + record.
    """
    for rec in records:
        if rec.base_model_id != base_model_id:
            raise ValueError(
                f"base_model_id mismatch on record {rec.task_id!r}: "
                f"expected {base_model_id!r}, got {rec.base_model_id!r}"
            )
        if rec.warm_start_adapter != warm_start_adapter:
            raise ValueError(
                f"warm_start_adapter mismatch on record {rec.task_id!r}: "
                f"expected {warm_start_adapter!r}, got {rec.warm_start_adapter!r}"
            )
        if rec.target_modules != target_modules:
            raise ValueError(
                f"target_modules mismatch on record {rec.task_id!r}: "
                f"expected {target_modules!r}, got {rec.target_modules!r}"
            )
        if rec.rank != rank:
            raise ValueError(
                f"rank mismatch on record {rec.task_id!r}: "
                f"expected {rank}, got {rec.rank}"
            )
        if rec.layer_indices != layer_indices:
            raise ValueError(
                f"layer_indices mismatch on record {rec.task_id!r}: "
                f"expected {layer_indices!r}, got {rec.layer_indices!r}"
            )


# asdict is re-exported for downstream debug/pprint use.
__all__ = [
    "SCHEMA_VERSION",
    "ReconstructionRecord",
    "ReconstructionManifest",
    "validate_homogeneity",
    "asdict",
]
