# Reconstruction Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 2–5 are parallel-safe (disjoint files); all others are sequential.

**Goal:** Ship a `model_training.reconstruction` subpackage that turns a directory tree of QLoRA oracle adapters (indexed by Rune's AdapterRegistry) into a T2L-compatible reconstruction-training manifest — the exact `{task_embs, lora_A, lora_B}` substrate that a future Sakana-style hypernetwork will learn to regress against.

**Architecture:** Read-only reducer. Query AdapterRegistry → load each adapter's safetensors → extract per-(module, layer) A/B matrices → embed each task's description → emit a manifest + per-task embeddings + optional z-score stats. Adapter weights stay where they are; only index artifacts are written. Mirrors `text-to-lora/src/hyper_llm_modulator/data.py::get_recon_train_data` shape so a downstream hypernetwork can consume the manifest without further transformation.

**Tech Stack:** Python 3.12, `safetensors`, `torch`, `sentence-transformers` (deferred), `sqlmodel` (via AdapterRegistry), `argparse`. GPU imports stay inside function bodies per INFRA-05.

---

## Source of Truth for the Target Shape

From T2L's `get_recon_train_data` (verified via WebFetch 2026-04-22), each oracle record must supply:

- `layer_indices: {module_name: LongTensor}` — sorted layer indices per module
- `lora_A: {module_name: Tensor[n_layers, rank, in_features]}`
- `lora_B: {module_name: Tensor[n_layers, out_features, rank]}`

PEFT stores A as `(rank, in_features)` and B as `(out_features, rank)` per layer, so stacking along dim 0 yields the T2L shapes **without** any transposes. Do not transpose B.

Task embedding: `torch.Tensor[1, hidden_size]` per task, keyed by task_id. For a real embedding model we use `sentence-transformers/all-mpnet-base-v2` (768-d) unless overridden; we also support a `None` embedding model, in which case we emit orthogonal one-hot vectors (matching T2L's `torch.eye` fallback).

## DeltaCoder-Relative Semantics (Critical)

Per `2026-04-22-pr-28-training-upgrade-fit-assessment.yaml` line 157–160, `trainer.save_model()` writes only the adapter delta, not a base-merged adapter. That means the A/B matrices we extract are deltas **on top of DeltaCoder**, not on top of raw Qwen3.5-9B. The manifest MUST record the warm-start adapter so downstream consumers load the correct coordinate system at train/inference time. See `manifest.ReconstructionManifest.warm_start_adapter`.

---

## File Structure

All new files. No edits to existing modules.

```
libs/model-training/src/model_training/reconstruction/
├── __init__.py               # empty (docstring only); consumers use explicit submodule imports
├── manifest.py               # ReconstructionRecord, ReconstructionManifest, JSON (de)serialization
├── extract.py                # extract_lora_ab_from_state_dict, load_adapter_as_record
├── registry_source.py        # iter_reconstruction_candidates: AdapterRegistry → list[AdapterRecord]
├── task_embeddings.py        # compute_task_embeddings, save_task_embeddings, load_task_embeddings
├── stats.py                  # compute_zscore_stats, save_zscore_stats
├── builder.py                # build_reconstruction_dataset (orchestrator)
└── cli.py                    # argparse + --dry-run + main()

libs/model-training/tests/
├── test_reconstruction_manifest.py
├── test_reconstruction_extract.py
├── test_reconstruction_registry_source.py
├── test_reconstruction_task_embeddings.py
├── test_reconstruction_stats.py
├── test_reconstruction_builder.py            # integration: 2-adapter fixture tree
└── test_reconstruction_cli.py
```

Each task owns one source file and its test file. No two tasks touch the same file. This makes Tasks 2–5 safe to run in parallel subagents.

---

## Task Dependency Graph

```
Task 1 (manifest.py) ─┬─► Task 2 (extract.py)         ─┐
                      ├─► Task 3 (task_embeddings.py)  ├─► Task 6 (builder.py) ─► Task 7 (cli.py) ─► Task 8 (integration)
                      ├─► Task 4 (registry_source.py)  │
                      └─► Task 5 (stats.py)            ─┘
```

Tasks 2, 3, 4, 5 have no runtime dependency on each other — only on manifest types from Task 1. Dispatch them as a parallel wave. Tasks 6–8 are sequential.

---

## Conventions for Every Task

- Use `uv run pytest <path>` for tests. Never bare `pytest` or `python`.
- Use `uv run ruff check` and `uv run mypy libs/model-training` before each commit.
- GPU imports (`torch`, `safetensors`, `sentence_transformers`) go inside function bodies, never at module top level. CPU-only CI must still `import model_training.reconstruction.*` without error.
- All new public functions get Google-style docstrings.
- Commit after each task with Conventional Commits style: `feat(reconstruction): <what>`.

---

## Task 1: Manifest Dataclasses + Serialization

**Files:**
- Create: `libs/model-training/src/model_training/reconstruction/__init__.py`
- Create: `libs/model-training/src/model_training/reconstruction/manifest.py`
- Test: `libs/model-training/tests/test_reconstruction_manifest.py`

**Acceptance:** `ReconstructionRecord` and `ReconstructionManifest` round-trip through JSON with lossless field preservation. `validate_homogeneity()` raises on mismatched `base_model_id`, `warm_start_adapter`, `rank`, `target_modules`, or `layer_indices`.

- [ ] **Step 1.1: Create the subpackage marker**

Create `libs/model-training/src/model_training/reconstruction/__init__.py` with content:

```python
"""Reconstruction-mode training dataset tooling for T2L-style hypernetworks.

Consumers import specific submodules directly:

    from model_training.reconstruction.builder import build_reconstruction_dataset
    from model_training.reconstruction.manifest import ReconstructionManifest
"""
```

- [ ] **Step 1.2: Write failing tests for ReconstructionRecord**

Create `libs/model-training/tests/test_reconstruction_manifest.py`:

```python
"""Tests for ReconstructionRecord / ReconstructionManifest JSON round-trip + validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_training.reconstruction.manifest import (
    ReconstructionManifest,
    ReconstructionRecord,
    validate_homogeneity,
)


def _rec(**overrides: object) -> ReconstructionRecord:
    defaults: dict[str, object] = {
        "task_id": "adapter-001",
        "adapter_path": "/adapters/adapter-001",
        "task_description": "fix off-by-one in list slicing",
        "base_model_id": "Qwen/Qwen3.5-9B",
        "warm_start_adapter": "danielcherubini/Qwen3.5-DeltaCoder-9B",
        "rank": 64,
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "layer_indices": tuple(range(32)),
        "created_at": "2026-04-22T00:00:00Z",
        "source_task_hash": "task-hash-001",
        "fitness_score": 0.82,
    }
    defaults.update(overrides)
    return ReconstructionRecord(**defaults)  # type: ignore[arg-type]


def test_record_roundtrips_via_to_dict_from_dict() -> None:
    original = _rec()
    round_tripped = ReconstructionRecord.from_dict(original.to_dict())
    assert round_tripped == original


def test_manifest_roundtrips_via_json(tmp_path: Path) -> None:
    manifest = ReconstructionManifest(
        schema_version=1,
        base_model_id="Qwen/Qwen3.5-9B",
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        rank=64,
        layer_indices=tuple(range(32)),
        task_embedding_model="sentence-transformers/all-mpnet-base-v2",
        task_embedding_dim=768,
        records=[_rec(task_id="a"), _rec(task_id="b")],
        zscore_stats_path=None,
        created_at="2026-04-22T00:00:00Z",
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ReconstructionManifest.load(path)
    assert loaded == manifest
    # Must be human-readable JSON.
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
    assert len(parsed["records"]) == 2


def test_validate_homogeneity_passes_for_matching_records() -> None:
    records = [_rec(task_id="a"), _rec(task_id="b")]
    validate_homogeneity(
        records,
        base_model_id="Qwen/Qwen3.5-9B",
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        rank=64,
        layer_indices=tuple(range(32)),
    )  # no raise


def test_validate_homogeneity_raises_on_base_mismatch() -> None:
    records = [_rec(task_id="a"), _rec(task_id="b", base_model_id="Qwen/Qwen2.5-Coder-7B")]
    with pytest.raises(ValueError, match="base_model_id"):
        validate_homogeneity(
            records,
            base_model_id="Qwen/Qwen3.5-9B",
            warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=64,
            layer_indices=tuple(range(32)),
        )


def test_validate_homogeneity_raises_on_rank_mismatch() -> None:
    records = [_rec(task_id="a"), _rec(task_id="b", rank=32)]
    with pytest.raises(ValueError, match="rank"):
        validate_homogeneity(
            records,
            base_model_id="Qwen/Qwen3.5-9B",
            warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=64,
            layer_indices=tuple(range(32)),
        )


def test_validate_homogeneity_raises_on_target_modules_mismatch() -> None:
    records = [
        _rec(task_id="a"),
        _rec(task_id="b", target_modules=("q_proj", "v_proj")),
    ]
    with pytest.raises(ValueError, match="target_modules"):
        validate_homogeneity(
            records,
            base_model_id="Qwen/Qwen3.5-9B",
            warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=64,
            layer_indices=tuple(range(32)),
        )


def test_manifest_module_is_cpu_importable() -> None:
    # No torch / safetensors at import time.
    import importlib

    mod = importlib.import_module("model_training.reconstruction.manifest")
    assert hasattr(mod, "ReconstructionRecord")
```

- [ ] **Step 1.3: Run the tests to confirm they fail**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_training.reconstruction.manifest'`

- [ ] **Step 1.4: Implement manifest.py**

Create `libs/model-training/src/model_training/reconstruction/manifest.py`:

```python
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
```

- [ ] **Step 1.5: Run the tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_manifest.py -v`
Expected: PASS — 7 tests.

- [ ] **Step 1.6: Lint + type-check**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction libs/model-training/tests/test_reconstruction_manifest.py`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction/manifest.py`
Expected: no errors.

- [ ] **Step 1.7: Commit**

```bash
git add libs/model-training/src/model_training/reconstruction/__init__.py \
        libs/model-training/src/model_training/reconstruction/manifest.py \
        libs/model-training/tests/test_reconstruction_manifest.py
git commit -m "feat(reconstruction): manifest dataclasses + JSON round-trip"
```

---

## Task 2: State Dict Extraction + Adapter Loading

**Parallel-safe with Tasks 3, 4, 5 after Task 1 lands.**

**Files:**
- Create: `libs/model-training/src/model_training/reconstruction/extract.py`
- Test: `libs/model-training/tests/test_reconstruction_extract.py`

**Acceptance:** Given a fabricated PEFT state_dict with keys like `base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight`, the extractor returns `{module: {"A": [n_layers, r, in], "B": [n_layers, out, r], "layer_indices": LongTensor}}` with values stacked in sorted layer-index order. Loading a real adapter dir (via safetensors) produces the same structure.

- [ ] **Step 2.1: Write failing tests**

Create `libs/model-training/tests/test_reconstruction_extract.py`:

```python
"""Tests for PEFT state_dict → T2L-shape extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _make_lora_key(layer: int, prefix: str, module: str, ab: str) -> str:
    return f"base_model.model.model.layers.{layer}.{prefix}.{module}.lora_{ab}.weight"


def _fabricate_state_dict(
    *,
    layers: tuple[int, ...],
    module_prefix: dict[str, str],  # module -> "self_attn" or "mlp"
    rank: int,
    in_features: int,
    out_features: int,
    dtype: "torch.dtype" = None,  # type: ignore[assignment]
) -> dict[str, "torch.Tensor"]:
    if dtype is None:
        dtype = torch.float32
    sd: dict[str, torch.Tensor] = {}
    for mod, prefix in module_prefix.items():
        for layer in layers:
            # PEFT layout: A is (rank, in_features); B is (out_features, rank).
            sd[_make_lora_key(layer, prefix, mod, "A")] = torch.randn(
                rank, in_features, dtype=dtype
            )
            sd[_make_lora_key(layer, prefix, mod, "B")] = torch.randn(
                out_features, rank, dtype=dtype
            )
    return sd


def test_extract_shapes_and_layer_order() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(2, 0, 1),  # unsorted on purpose
        module_prefix={"q_proj": "self_attn", "gate_proj": "mlp"},
        rank=8,
        in_features=128,
        out_features=256,
    )
    out = extract_lora_ab_from_state_dict(
        sd, target_modules=("q_proj", "gate_proj")
    )
    assert set(out) == {"q_proj", "gate_proj"}
    for mod in ("q_proj", "gate_proj"):
        assert out[mod]["A"].shape == (3, 8, 128)
        assert out[mod]["B"].shape == (3, 256, 8)
        assert out[mod]["layer_indices"].tolist() == [0, 1, 2]


def test_extract_detects_layer_indices_per_module() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(0, 1, 2, 3),
        module_prefix={"q_proj": "self_attn"},
        rank=4,
        in_features=64,
        out_features=64,
    )
    out = extract_lora_ab_from_state_dict(sd, target_modules=("q_proj",))
    assert out["q_proj"]["layer_indices"].tolist() == [0, 1, 2, 3]


def test_extract_raises_when_module_has_no_keys() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(0, 1),
        module_prefix={"q_proj": "self_attn"},
        rank=4,
        in_features=32,
        out_features=32,
    )
    with pytest.raises(ValueError, match="no .* keys for module 'v_proj'"):
        extract_lora_ab_from_state_dict(sd, target_modules=("q_proj", "v_proj"))


def test_extract_raises_when_a_b_layer_sets_disagree() -> None:
    from model_training.reconstruction.extract import extract_lora_ab_from_state_dict

    sd = _fabricate_state_dict(
        layers=(0, 1),
        module_prefix={"q_proj": "self_attn"},
        rank=4,
        in_features=32,
        out_features=32,
    )
    del sd[_make_lora_key(1, "self_attn", "q_proj", "B")]
    with pytest.raises(ValueError, match="lora_A .* lora_B .* mismatch"):
        extract_lora_ab_from_state_dict(sd, target_modules=("q_proj",))


def test_load_adapter_as_record(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from model_training.reconstruction.extract import load_adapter_as_record

    sd = _fabricate_state_dict(
        layers=(0, 1),
        module_prefix={"q_proj": "self_attn", "v_proj": "self_attn"},
        rank=4,
        in_features=32,
        out_features=32,
    )
    adapter_dir = tmp_path / "adapter-abc"
    adapter_dir.mkdir()
    save_file(sd, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "danielcherubini/Qwen3.5-DeltaCoder-9B",
                "r": 4,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM",
            }
        )
    )
    record_kwargs = load_adapter_as_record(
        adapter_dir,
        task_id="abc",
        task_description="some description",
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        base_model_id_override="Qwen/Qwen3.5-9B",
        created_at="2026-04-22T00:00:00Z",
    )
    assert record_kwargs["rank"] == 4
    assert record_kwargs["target_modules"] == ("q_proj", "v_proj")
    assert record_kwargs["layer_indices"] == (0, 1)
    assert record_kwargs["base_model_id"] == "Qwen/Qwen3.5-9B"
    assert record_kwargs["warm_start_adapter"] == "danielcherubini/Qwen3.5-DeltaCoder-9B"


def test_extract_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.extract")
    assert hasattr(mod, "extract_lora_ab_from_state_dict")
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_extract.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2.3: Implement extract.py**

Create `libs/model-training/src/model_training/reconstruction/extract.py`:

```python
"""Extract per-module LoRA A/B matrices from PEFT state_dicts + adapter dirs.

Mirrors the key-parse logic in T2L's
``hyper_llm_modulator/data.py::get_recon_train_data``. PEFT stores A as
``(rank, in_features)`` and B as ``(out_features, rank)`` per layer; we stack
along a new layer axis (dim 0) without transposing.

Torch is imported inside function bodies (INFRA-05).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Example PEFT keys this regex must match:
#   base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight
#   base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight
_KEY_RE = re.compile(
    r"^base_model\.model\.model\.layers\."
    r"(?P<layer>\d+)\."
    r"(?P<prefix>[^.]+)\."
    r"(?P<module>[^.]+)\."
    r"lora_(?P<ab>[AB])\.weight$"
)


def extract_lora_ab_from_state_dict(
    state_dict: dict[str, Any],
    *,
    target_modules: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Return ``{module: {"A": Tensor[L, r, in], "B": Tensor[L, out, r], "layer_indices": LongTensor[L]}}``.

    Args:
        state_dict: Dict mapping PEFT key → tensor. Both ``torch.Tensor``
            and safetensors-loaded tensors work.
        target_modules: Module names to extract. Must appear at the module
            slot of the PEFT key (e.g. ``"q_proj"``, ``"gate_proj"``).

    Raises:
        ValueError: If any requested module has zero matching keys, or if
            the A and B layer sets disagree for a module.
    """
    import torch  # noqa: PLC0415

    # Collect A/B per (module, layer) first, then assemble stacks.
    a_by_mod_layer: dict[str, dict[int, torch.Tensor]] = {m: {} for m in target_modules}
    b_by_mod_layer: dict[str, dict[int, torch.Tensor]] = {m: {} for m in target_modules}

    for key, tensor in state_dict.items():
        match = _KEY_RE.match(key)
        if match is None:
            continue
        module = match.group("module")
        if module not in a_by_mod_layer:
            continue
        layer = int(match.group("layer"))
        if match.group("ab") == "A":
            a_by_mod_layer[module][layer] = tensor
        else:
            b_by_mod_layer[module][layer] = tensor

    out: dict[str, dict[str, Any]] = {}
    for module in target_modules:
        a_layers = a_by_mod_layer[module]
        b_layers = b_by_mod_layer[module]
        if not a_layers and not b_layers:
            raise ValueError(
                f"no lora_A/lora_B keys for module '{module}' in state_dict"
            )
        if set(a_layers) != set(b_layers):
            only_a = sorted(set(a_layers) - set(b_layers))
            only_b = sorted(set(b_layers) - set(a_layers))
            raise ValueError(
                f"lora_A and lora_B layer sets mismatch for '{module}': "
                f"only_A={only_a}, only_B={only_b}"
            )
        sorted_layers = sorted(a_layers.keys())
        a_stack = torch.stack([a_layers[i] for i in sorted_layers], dim=0)
        b_stack = torch.stack([b_layers[i] for i in sorted_layers], dim=0)
        out[module] = {
            "A": a_stack,
            "B": b_stack,
            "layer_indices": torch.tensor(sorted_layers, dtype=torch.long),
        }
    return out


def load_adapter_state_dict(adapter_dir: Path) -> dict[str, Any]:
    """Load ``adapter_model.safetensors`` from a PEFT adapter directory."""
    from safetensors.torch import load_file  # noqa: PLC0415

    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.is_file():
        raise FileNotFoundError(f"missing adapter_model.safetensors: {st_path}")
    return load_file(str(st_path))


def load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    """Load ``adapter_config.json`` from a PEFT adapter directory."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing adapter_config.json: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_adapter_as_record(
    adapter_dir: Path,
    *,
    task_id: str,
    task_description: str,
    warm_start_adapter: str | None,
    base_model_id_override: str | None,
    created_at: str,
    source_task_hash: str | None = None,
    fitness_score: float | None = None,
) -> dict[str, Any]:
    """Read an adapter dir and return kwargs for ``ReconstructionRecord(**kwargs)``.

    The A/B tensors themselves are NOT returned — they live on disk where the
    trainer can stream them. Only shape / identity metadata flows through.

    Args:
        adapter_dir: Directory containing ``adapter_model.safetensors`` and
            ``adapter_config.json``.
        task_id: Manifest-stable id (usually ``AdapterRecord.id``).
        task_description: Text used to compute the task embedding.
        warm_start_adapter: Warm-start adapter path/repo, or None. Stored
            verbatim on the record — downstream consumers must honor this.
        base_model_id_override: If set, overrides the ``base_model_name_or_path``
            from ``adapter_config.json`` (e.g., when the config records the
            warm-start adapter instead of the true base).
        created_at: ISO-8601 UTC timestamp.
        source_task_hash: Optional dedup key.
        fitness_score: Optional evaluation score.

    Returns:
        Dict suitable for ``ReconstructionRecord(**returned_dict)``.
    """
    cfg = load_adapter_config(adapter_dir)
    target_modules: tuple[str, ...] = tuple(cfg["target_modules"])
    rank = int(cfg["r"])

    state_dict = load_adapter_state_dict(adapter_dir)
    per_mod = extract_lora_ab_from_state_dict(state_dict, target_modules=target_modules)

    # All modules share the same layer set by construction (validate_homogeneity).
    first_mod = target_modules[0]
    layer_indices = tuple(int(i) for i in per_mod[first_mod]["layer_indices"].tolist())

    base_from_cfg = str(cfg.get("base_model_name_or_path", ""))
    base_model_id = base_model_id_override or base_from_cfg
    if not base_model_id:
        raise ValueError(
            f"adapter {adapter_dir} has no base_model_name_or_path and no override"
        )

    return {
        "task_id": task_id,
        "adapter_path": str(adapter_dir.resolve()),
        "task_description": task_description,
        "base_model_id": base_model_id,
        "warm_start_adapter": warm_start_adapter,
        "rank": rank,
        "target_modules": target_modules,
        "layer_indices": layer_indices,
        "created_at": created_at,
        "source_task_hash": source_task_hash,
        "fitness_score": fitness_score,
    }


__all__ = [
    "extract_lora_ab_from_state_dict",
    "load_adapter_state_dict",
    "load_adapter_config",
    "load_adapter_as_record",
]
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_extract.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 2.5: Lint + type-check**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction/extract.py libs/model-training/tests/test_reconstruction_extract.py`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction/extract.py`
Expected: no errors.

- [ ] **Step 2.6: Commit**

```bash
git add libs/model-training/src/model_training/reconstruction/extract.py \
        libs/model-training/tests/test_reconstruction_extract.py
git commit -m "feat(reconstruction): PEFT state_dict → per-module A/B extraction"
```

---

## Task 3: Task Embeddings

**Parallel-safe with Tasks 2, 4, 5 after Task 1 lands.**

**Files:**
- Create: `libs/model-training/src/model_training/reconstruction/task_embeddings.py`
- Test: `libs/model-training/tests/test_reconstruction_task_embeddings.py`

**Acceptance:** `compute_task_embeddings({"a": "desc-a", "b": "desc-b"}, model=None)` returns an orthogonal one-hot matrix (2×2) keyed by task_id. With `model=<fake>`, returns whatever the fake model emits. Save + load round-trip preserves the dict keys and tensor values. The default model id constant is `"sentence-transformers/all-mpnet-base-v2"` with dim 768.

- [ ] **Step 3.1: Write failing tests**

Create `libs/model-training/tests/test_reconstruction_task_embeddings.py`:

```python
"""Tests for task embedding computation + persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def test_one_hot_fallback_when_model_is_none() -> None:
    from model_training.reconstruction.task_embeddings import compute_task_embeddings

    embs = compute_task_embeddings(
        {"a": "desc-a", "b": "desc-b", "c": "desc-c"},
        model=None,
    )
    assert set(embs) == {"a", "b", "c"}
    # each embedding is (1, 3) with a single 1.0
    stacked = torch.cat([embs[k] for k in ("a", "b", "c")], dim=0)
    assert stacked.shape == (3, 3)
    assert torch.allclose(stacked @ stacked.T, torch.eye(3))


def test_uses_provided_encoder() -> None:
    from model_training.reconstruction.task_embeddings import compute_task_embeddings

    class _FakeEncoder:
        def encode(
            self, texts: list[str], convert_to_tensor: bool = True, **_: Any
        ) -> torch.Tensor:
            return torch.tensor(
                [[float(len(t)), float(len(t)) * 2] for t in texts],
                dtype=torch.float32,
            )

    embs = compute_task_embeddings({"a": "xy", "b": "abc"}, model=_FakeEncoder())
    assert embs["a"].shape == (1, 2)
    assert torch.allclose(embs["a"], torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(embs["b"], torch.tensor([[3.0, 6.0]]))


def test_roundtrips_via_save_load(tmp_path: Path) -> None:
    from model_training.reconstruction.task_embeddings import (
        load_task_embeddings,
        save_task_embeddings,
    )

    embs = {"a": torch.randn(1, 8), "b": torch.randn(1, 8)}
    path = tmp_path / "task_embeddings.pt"
    save_task_embeddings(embs, path)
    loaded = load_task_embeddings(path)
    assert set(loaded) == {"a", "b"}
    for k in embs:
        assert torch.allclose(loaded[k], embs[k])


def test_default_model_id_constant() -> None:
    from model_training.reconstruction.task_embeddings import (
        DEFAULT_EMBEDDING_DIM,
        DEFAULT_EMBEDDING_MODEL,
    )

    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-mpnet-base-v2"
    assert DEFAULT_EMBEDDING_DIM == 768


def test_task_embeddings_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.task_embeddings")
    assert hasattr(mod, "compute_task_embeddings")
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_task_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement task_embeddings.py**

Create `libs/model-training/src/model_training/reconstruction/task_embeddings.py`:

```python
"""Task embedding computation + persistence for reconstruction datasets.

Mirrors T2L's ``get_task_embs`` (see hyper_llm_modulator/data.py). Default
encoder is ``sentence-transformers/all-mpnet-base-v2`` (768-d). A ``None``
model falls back to one-hot orthogonal vectors.

sentence-transformers is imported inside ``load_default_encoder`` (INFRA-05).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBEDDING_DIM = 768


class _Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


def load_default_encoder(
    model_id: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"
) -> _Encoder:
    """Instantiate a sentence-transformers SentenceTransformer encoder.

    Args:
        model_id: HF repo id.
        device: ``"cpu"`` or ``"cuda"``.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(model_id, device=device)


def compute_task_embeddings(
    descriptions: dict[str, str],
    *,
    model: _Encoder | None,
) -> dict[str, Any]:
    """Compute per-task embeddings, returning ``{task_id: Tensor[1, dim]}``.

    When ``model`` is None, emits orthogonal one-hot vectors of dim
    ``len(descriptions)``. This matches T2L's fallback in ``get_task_embs``
    and guarantees the hypernetwork can still discriminate tasks even when
    no encoder is available.
    """
    import torch  # noqa: PLC0415

    task_ids = list(descriptions.keys())
    n = len(task_ids)

    if model is None:
        logger.info("compute_task_embeddings: no encoder → one-hot fallback (dim=%d)", n)
        eye = torch.eye(n, dtype=torch.float32)
        return {tid: eye[i].unsqueeze(0) for i, tid in enumerate(task_ids)}

    texts = [descriptions[tid] for tid in task_ids]
    encoded = model.encode(texts, convert_to_tensor=True)
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.as_tensor(encoded)
    encoded = encoded.float().cpu()
    if encoded.ndim != 2 or encoded.shape[0] != n:
        raise ValueError(
            f"encoder returned shape {tuple(encoded.shape)}, expected ({n}, dim)"
        )
    return {tid: encoded[i].unsqueeze(0).contiguous() for i, tid in enumerate(task_ids)}


def save_task_embeddings(embeddings: dict[str, Any], path: Path) -> None:
    """Persist ``{task_id: Tensor}`` as a ``torch.save`` file."""
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, str(path))


def load_task_embeddings(path: Path) -> dict[str, Any]:
    """Inverse of ``save_task_embeddings``."""
    import torch  # noqa: PLC0415

    loaded = torch.load(str(path), weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected dict in {path}, got {type(loaded)!r}")
    return loaded


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIM",
    "compute_task_embeddings",
    "load_default_encoder",
    "save_task_embeddings",
    "load_task_embeddings",
]
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_task_embeddings.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 3.5: Lint + type-check**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction/task_embeddings.py libs/model-training/tests/test_reconstruction_task_embeddings.py`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction/task_embeddings.py`
Expected: no errors.

- [ ] **Step 3.6: Commit**

```bash
git add libs/model-training/src/model_training/reconstruction/task_embeddings.py \
        libs/model-training/tests/test_reconstruction_task_embeddings.py
git commit -m "feat(reconstruction): task embedding computation + persistence"
```

---

## Task 4: Registry Source

**Parallel-safe with Tasks 2, 3, 5 after Task 1 lands.**

**Files:**
- Create: `libs/model-training/src/model_training/reconstruction/registry_source.py`
- Test: `libs/model-training/tests/test_reconstruction_registry_source.py`

**Acceptance:** `iter_reconstruction_candidates(registry, task_type=..., min_fitness=..., sources=...)` returns a list of `AdapterRecord` filtered by optional criteria, excluding archived adapters. Uncompilable or non-existent file_path entries are skipped with a warning (don't crash the build).

- [ ] **Step 4.1: Write failing tests**

Create `libs/model-training/tests/test_reconstruction_registry_source.py`:

```python
"""Tests for iter_reconstruction_candidates against a real AdapterRegistry."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from adapter_registry.models import AdapterRecord
from adapter_registry.registry import AdapterRegistry
from sqlalchemy import create_engine


@pytest.fixture
def registry(tmp_path: Path) -> AdapterRegistry:
    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    return AdapterRegistry(engine=engine)


def _populate(
    registry: AdapterRegistry,
    make_adapter_record: Callable[..., AdapterRecord],
    tmp_path: Path,
) -> None:
    for i, (task_type, fitness, source, archived) in enumerate(
        [
            ("bug-fix", 0.8, "distillation", False),
            ("bug-fix", 0.4, "distillation", False),
            ("bug-fix", 0.9, "evolution", False),
            ("refactor", 0.7, "distillation", False),
            ("bug-fix", 0.95, "distillation", True),  # archived
        ]
    ):
        adapter_dir = tmp_path / f"adapter-{i:03d}"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"stub")
        (adapter_dir / "adapter_config.json").write_text("{}")
        rec = make_adapter_record(
            id=f"adapter-{i:03d}",
            task_type=task_type,
            fitness_score=fitness,
            source=source,
            is_archived=archived,
            file_path=str(adapter_dir),
        )
        registry.store(rec)
        if archived:
            registry.archive(rec.id)


def test_filters_archived(
    registry: AdapterRegistry,
    make_adapter_record: Callable[..., AdapterRecord],
    tmp_path: Path,
) -> None:
    from model_training.reconstruction.registry_source import (
        iter_reconstruction_candidates,
    )

    _populate(registry, make_adapter_record, tmp_path)
    results = iter_reconstruction_candidates(registry)
    ids = {r.id for r in results}
    assert "adapter-004" not in ids  # archived
    assert len(ids) == 4


def test_filters_by_task_type(
    registry: AdapterRegistry,
    make_adapter_record: Callable[..., AdapterRecord],
    tmp_path: Path,
) -> None:
    from model_training.reconstruction.registry_source import (
        iter_reconstruction_candidates,
    )

    _populate(registry, make_adapter_record, tmp_path)
    results = iter_reconstruction_candidates(registry, task_type="refactor")
    assert {r.id for r in results} == {"adapter-003"}


def test_filters_by_min_fitness(
    registry: AdapterRegistry,
    make_adapter_record: Callable[..., AdapterRecord],
    tmp_path: Path,
) -> None:
    from model_training.reconstruction.registry_source import (
        iter_reconstruction_candidates,
    )

    _populate(registry, make_adapter_record, tmp_path)
    results = iter_reconstruction_candidates(registry, min_fitness=0.75)
    ids = {r.id for r in results}
    assert ids == {"adapter-000", "adapter-002"}
    # adapter-004 has fitness 0.95 but is archived, so excluded.


def test_filters_by_sources(
    registry: AdapterRegistry,
    make_adapter_record: Callable[..., AdapterRecord],
    tmp_path: Path,
) -> None:
    from model_training.reconstruction.registry_source import (
        iter_reconstruction_candidates,
    )

    _populate(registry, make_adapter_record, tmp_path)
    results = iter_reconstruction_candidates(registry, sources=("evolution",))
    assert {r.id for r in results} == {"adapter-002"}


def test_drops_adapters_with_missing_file_path(
    registry: AdapterRegistry,
    make_adapter_record: Callable[..., AdapterRecord],
    tmp_path: Path,
) -> None:
    from model_training.reconstruction.registry_source import (
        iter_reconstruction_candidates,
    )

    rec = make_adapter_record(
        id="adapter-missing", file_path=str(tmp_path / "does-not-exist")
    )
    registry.store(rec)
    results = iter_reconstruction_candidates(registry)
    assert all(r.id != "adapter-missing" for r in results)


def test_registry_source_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.registry_source")
    assert hasattr(mod, "iter_reconstruction_candidates")
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_registry_source.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4.3: Implement registry_source.py**

Create `libs/model-training/src/model_training/reconstruction/registry_source.py`:

```python
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
            r for r in records if r.fitness_score is not None and r.fitness_score >= min_fitness
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
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_registry_source.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 4.5: Lint + type-check**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction/registry_source.py libs/model-training/tests/test_reconstruction_registry_source.py`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction/registry_source.py`
Expected: no errors.

- [ ] **Step 4.6: Commit**

```bash
git add libs/model-training/src/model_training/reconstruction/registry_source.py \
        libs/model-training/tests/test_reconstruction_registry_source.py
git commit -m "feat(reconstruction): registry filter for reconstruction candidates"
```

---

## Task 5: Z-Score Stats

**Parallel-safe with Tasks 2, 3, 4 after Task 1 lands.**

**Files:**
- Create: `libs/model-training/src/model_training/reconstruction/stats.py`
- Test: `libs/model-training/tests/test_reconstruction_stats.py`

**Acceptance:** Given an iterator of per-record `{module: {"A": tensor, "B": tensor}}` dicts, `compute_zscore_stats` returns `{module: {"avg_A", "std_A", "avg_B", "std_B"}}` with `std` values floored at `1e-6`. Saves + loads through `.pt` losslessly. Matches T2L's `std_recon_target + 1e-10` + `pred_z_score` normalization semantics (stats are computed across tasks, not within a task).

- [ ] **Step 5.1: Write failing tests**

Create `libs/model-training/tests/test_reconstruction_stats.py`:

```python
"""Tests for across-corpus z-score statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _per_record(a: "torch.Tensor", b: "torch.Tensor") -> dict[str, dict[str, "torch.Tensor"]]:
    return {"q_proj": {"A": a, "B": b}}


def test_mean_and_std_shapes_match_single_record_shape() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    a = torch.randn(4, 8, 16)  # (n_layers, r, in_features)
    b = torch.randn(4, 32, 8)
    stats = compute_zscore_stats([_per_record(a, b)])
    assert stats["q_proj"]["avg_A"].shape == (4, 8, 16)
    assert stats["q_proj"]["std_A"].shape == (4, 8, 16)
    assert stats["q_proj"]["avg_B"].shape == (4, 32, 8)
    assert stats["q_proj"]["std_B"].shape == (4, 32, 8)


def test_mean_matches_elementwise_average_across_records() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    a1 = torch.ones(2, 2, 4)
    a2 = torch.full((2, 2, 4), 3.0)
    b1 = torch.ones(2, 4, 2)
    b2 = torch.full((2, 4, 2), 3.0)
    stats = compute_zscore_stats([_per_record(a1, b1), _per_record(a2, b2)])
    assert torch.allclose(stats["q_proj"]["avg_A"], torch.full((2, 2, 4), 2.0))
    assert torch.allclose(stats["q_proj"]["avg_B"], torch.full((2, 4, 2), 2.0))


def test_std_floored_to_minimum() -> None:
    from model_training.reconstruction.stats import (
        STD_FLOOR,
        compute_zscore_stats,
    )

    a = torch.zeros(2, 2, 4)  # zero variance
    b = torch.zeros(2, 4, 2)
    stats = compute_zscore_stats([_per_record(a, b), _per_record(a, b)])
    assert torch.all(stats["q_proj"]["std_A"] >= STD_FLOOR)
    assert torch.all(stats["q_proj"]["std_B"] >= STD_FLOOR)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    from model_training.reconstruction.stats import (
        compute_zscore_stats,
        load_zscore_stats,
        save_zscore_stats,
    )

    a = torch.randn(3, 4, 8)
    b = torch.randn(3, 16, 4)
    stats = compute_zscore_stats([_per_record(a, b), _per_record(a * 0.5, b * 2)])
    path = tmp_path / "zscore.pt"
    save_zscore_stats(stats, path)
    loaded = load_zscore_stats(path)
    for module in stats:
        for key in ("avg_A", "std_A", "avg_B", "std_B"):
            assert torch.allclose(loaded[module][key], stats[module][key])


def test_compute_raises_on_empty_input() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    with pytest.raises(ValueError, match="at least one"):
        compute_zscore_stats([])


def test_compute_raises_on_inconsistent_modules() -> None:
    from model_training.reconstruction.stats import compute_zscore_stats

    a = torch.randn(2, 2, 4)
    b = torch.randn(2, 4, 2)
    r1 = {"q_proj": {"A": a, "B": b}}
    r2 = {"v_proj": {"A": a, "B": b}}
    with pytest.raises(ValueError, match="inconsistent modules"):
        compute_zscore_stats([r1, r2])


def test_stats_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.stats")
    assert hasattr(mod, "compute_zscore_stats")
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5.3: Implement stats.py**

Create `libs/model-training/src/model_training/reconstruction/stats.py`:

```python
"""Across-corpus z-score statistics for reconstruction targets.

Mirrors T2L's ``std_recon_target + 1e-10`` normalization (see
``hyper_llm_modulator/recon_trainer.py``). Stats are computed element-wise
across the task dimension so a downstream hypernetwork with ``pred_z_score``
can normalize oracle targets before the L1 loss.

Inputs are dicts of ``{module: {"A": Tensor[n_layers, r, in], "B": Tensor[n_layers, out, r]}}``
as produced by ``extract.extract_lora_ab_from_state_dict``. All records must
share the same module set and per-module tensor shapes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

STD_FLOOR = 1e-6


def compute_zscore_stats(
    per_record_tensors: Iterable[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Compute element-wise ``{avg_A, std_A, avg_B, std_B}`` per module.

    Args:
        per_record_tensors: Iterable of per-record dicts
            ``{module: {"A": Tensor, "B": Tensor}}``. At least one record
            required; all records must share the same module keys and
            per-module tensor shapes.

    Returns:
        ``{module: {"avg_A": Tensor, "std_A": Tensor, "avg_B": Tensor, "std_B": Tensor}}``.
        ``std`` tensors are floored to ``STD_FLOOR`` for numerical safety.

    Raises:
        ValueError: On empty iterable or inconsistent module sets across records.
    """
    import torch  # noqa: PLC0415

    records = list(per_record_tensors)
    if not records:
        raise ValueError("compute_zscore_stats requires at least one record")

    reference_modules = set(records[0].keys())
    for i, rec in enumerate(records[1:], start=1):
        if set(rec.keys()) != reference_modules:
            raise ValueError(
                f"inconsistent modules at record {i}: "
                f"expected {sorted(reference_modules)}, got {sorted(rec.keys())}"
            )

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for module in reference_modules:
        a_stack = torch.stack([rec[module]["A"] for rec in records], dim=0).float()
        b_stack = torch.stack([rec[module]["B"] for rec in records], dim=0).float()
        # n_records > 1 → use population std (unbiased=False) so the stat is
        # finite even with 2 samples. Downstream z-score normalization is
        # what matters; tiny-N variance is bounded by STD_FLOOR anyway.
        avg_a = a_stack.mean(dim=0)
        avg_b = b_stack.mean(dim=0)
        std_a = a_stack.std(dim=0, unbiased=False).clamp_min(STD_FLOOR)
        std_b = b_stack.std(dim=0, unbiased=False).clamp_min(STD_FLOOR)
        stats[module] = {
            "avg_A": avg_a,
            "std_A": std_a,
            "avg_B": avg_b,
            "std_B": std_b,
        }
    return stats


def save_zscore_stats(stats: dict[str, dict[str, Any]], path: Path) -> None:
    """Persist the stats dict via ``torch.save``."""
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, str(path))


def load_zscore_stats(path: Path) -> dict[str, dict[str, Any]]:
    """Inverse of ``save_zscore_stats``."""
    import torch  # noqa: PLC0415

    loaded = torch.load(str(path), weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected dict in {path}, got {type(loaded)!r}")
    return loaded


__all__ = ["STD_FLOOR", "compute_zscore_stats", "save_zscore_stats", "load_zscore_stats"]
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_stats.py -v`
Expected: PASS — 7 tests.

- [ ] **Step 5.5: Lint + type-check**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction/stats.py libs/model-training/tests/test_reconstruction_stats.py`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction/stats.py`
Expected: no errors.

- [ ] **Step 5.6: Commit**

```bash
git add libs/model-training/src/model_training/reconstruction/stats.py \
        libs/model-training/tests/test_reconstruction_stats.py
git commit -m "feat(reconstruction): across-corpus z-score statistics"
```

---

## Task 6: Builder Orchestrator

**Sequential. Requires Tasks 1–5 to be merged.**

**Files:**
- Create: `libs/model-training/src/model_training/reconstruction/builder.py`
- Test: `libs/model-training/tests/test_reconstruction_builder.py`

**Acceptance:** Given a registry with two adapter directories, a callable returning task descriptions, and an output directory, `build_reconstruction_dataset` emits `manifest.json` + `task_embeddings.pt` + optionally `zscore_stats.pt`. Writes a valid manifest referencing the adapter paths. Raises on homogeneity violations before any file is written.

- [ ] **Step 6.1: Write failing tests**

Create `libs/model-training/tests/test_reconstruction_builder.py`:

```python
"""Integration tests for build_reconstruction_dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from adapter_registry.models import AdapterRecord
from adapter_registry.registry import AdapterRegistry
from sqlalchemy import create_engine

torch = pytest.importorskip("torch")


def _write_fake_adapter(
    adapter_dir: Path,
    *,
    base_model_name_or_path: str,
    rank: int,
    target_modules: list[str],
    layer_indices: list[int],
    in_features: int = 16,
    out_features: int = 16,
) -> None:
    from safetensors.torch import save_file

    sd: dict[str, torch.Tensor] = {}
    for mod in target_modules:
        prefix = "self_attn" if mod in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
        for layer in layer_indices:
            sd[
                f"base_model.model.model.layers.{layer}.{prefix}.{mod}.lora_A.weight"
            ] = torch.randn(rank, in_features)
            sd[
                f"base_model.model.model.layers.{layer}.{prefix}.{mod}.lora_B.weight"
            ] = torch.randn(out_features, rank)
    adapter_dir.mkdir(parents=True)
    save_file(sd, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model_name_or_path,
                "r": rank,
                "target_modules": target_modules,
                "task_type": "CAUSAL_LM",
            }
        )
    )


@pytest.fixture
def populated_registry(
    tmp_path: Path, make_adapter_record: Callable[..., AdapterRecord]
) -> tuple[AdapterRegistry, Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    registry = AdapterRegistry(engine=engine)
    for i in range(2):
        adapter_dir = tmp_path / f"adapter-{i}"
        _write_fake_adapter(
            adapter_dir,
            base_model_name_or_path="danielcherubini/Qwen3.5-DeltaCoder-9B",
            rank=4,
            target_modules=["q_proj", "v_proj"],
            layer_indices=[0, 1],
        )
        registry.store(
            make_adapter_record(
                id=f"adapter-{i}",
                rank=4,
                file_path=str(adapter_dir),
                base_model_id="Qwen/Qwen3.5-9B",
                fitness_score=0.5 + 0.1 * i,
            )
        )
    return registry, tmp_path


def test_builds_manifest_plus_embeddings_plus_stats(
    populated_registry: tuple[AdapterRegistry, Path],
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset
    from model_training.reconstruction.manifest import ReconstructionManifest

    registry, base_dir = populated_registry
    out_dir = base_dir / "recon_ds"

    def describe(rec: AdapterRecord) -> str:
        return f"task-description-for-{rec.id}"

    build_reconstruction_dataset(
        registry=registry,
        out_dir=out_dir,
        task_description_fn=describe,
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        base_model_id_override="Qwen/Qwen3.5-9B",
        emb_model=None,  # one-hot fallback
        compute_zscore=True,
    )

    manifest_path = out_dir / "manifest.json"
    embeddings_path = out_dir / "task_embeddings.pt"
    stats_path = out_dir / "zscore_stats.pt"
    assert manifest_path.is_file()
    assert embeddings_path.is_file()
    assert stats_path.is_file()

    manifest = ReconstructionManifest.load(manifest_path)
    assert manifest.base_model_id == "Qwen/Qwen3.5-9B"
    assert manifest.warm_start_adapter == "danielcherubini/Qwen3.5-DeltaCoder-9B"
    assert manifest.rank == 4
    assert manifest.target_modules == ("q_proj", "v_proj")
    assert manifest.layer_indices == (0, 1)
    assert manifest.task_embedding_model is None
    assert manifest.task_embedding_dim == 2  # one-hot over 2 tasks
    assert {r.task_id for r in manifest.records} == {"adapter-0", "adapter-1"}
    assert manifest.zscore_stats_path == str(stats_path.resolve())


def test_skips_zscore_when_disabled(
    populated_registry: tuple[AdapterRegistry, Path],
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset
    from model_training.reconstruction.manifest import ReconstructionManifest

    registry, base_dir = populated_registry
    out_dir = base_dir / "recon_ds_no_stats"

    build_reconstruction_dataset(
        registry=registry,
        out_dir=out_dir,
        task_description_fn=lambda rec: rec.id,
        warm_start_adapter=None,
        base_model_id_override="Qwen/Qwen3.5-9B",
        emb_model=None,
        compute_zscore=False,
    )

    manifest = ReconstructionManifest.load(out_dir / "manifest.json")
    assert manifest.zscore_stats_path is None
    assert not (out_dir / "zscore_stats.pt").exists()


def test_raises_on_heterogeneous_ranks(
    tmp_path: Path, make_adapter_record: Callable[..., AdapterRecord]
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset

    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    registry = AdapterRegistry(engine=engine)
    for i, rank in enumerate((4, 8)):
        adapter_dir = tmp_path / f"adapter-{i}"
        _write_fake_adapter(
            adapter_dir,
            base_model_name_or_path="danielcherubini/Qwen3.5-DeltaCoder-9B",
            rank=rank,
            target_modules=["q_proj"],
            layer_indices=[0, 1],
        )
        registry.store(
            make_adapter_record(
                id=f"adapter-{i}", rank=rank, file_path=str(adapter_dir)
            )
        )
    with pytest.raises(ValueError, match="rank"):
        build_reconstruction_dataset(
            registry=registry,
            out_dir=tmp_path / "out",
            task_description_fn=lambda rec: rec.id,
            warm_start_adapter=None,
            base_model_id_override="Qwen/Qwen3.5-9B",
            emb_model=None,
            compute_zscore=False,
        )
    # out_dir must not exist — we fail before emitting.
    assert not (tmp_path / "out").exists()


def test_raises_on_no_candidates(
    tmp_path: Path,
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset

    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    registry = AdapterRegistry(engine=engine)
    with pytest.raises(ValueError, match="no candidates"):
        build_reconstruction_dataset(
            registry=registry,
            out_dir=tmp_path / "out",
            task_description_fn=lambda rec: rec.id,
            warm_start_adapter=None,
            base_model_id_override="Qwen/Qwen3.5-9B",
            emb_model=None,
            compute_zscore=False,
        )


def test_builder_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.builder")
    assert hasattr(mod, "build_reconstruction_dataset")
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 6.3: Implement builder.py**

Create `libs/model-training/src/model_training/reconstruction/builder.py`:

```python
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
        task_type, min_fitness, sources: Passed through to
            ``iter_reconstruction_candidates``.
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
            "no candidates found in registry (check task_type/min_fitness/sources filters)"
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
```

- [ ] **Step 6.4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_builder.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 6.5: Lint + type-check**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction/builder.py libs/model-training/tests/test_reconstruction_builder.py`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction/builder.py`
Expected: no errors.

- [ ] **Step 6.6: Commit**

```bash
git add libs/model-training/src/model_training/reconstruction/builder.py \
        libs/model-training/tests/test_reconstruction_builder.py
git commit -m "feat(reconstruction): registry → manifest orchestrator"
```

---

## Task 7: CLI with Dry-Run

**Sequential. Requires Task 6 merged.**

**Files:**
- Create: `libs/model-training/src/model_training/reconstruction/cli.py`
- Test: `libs/model-training/tests/test_reconstruction_cli.py`

**Acceptance:** `uv run python -m model_training.reconstruction.cli --database-url sqlite:///... --out-dir ... --warm-start deltacoder --base-model qwen3.5-9b --dry-run` prints resolved kwargs as JSON without importing torch. Without `--dry-run`, it calls `build_reconstruction_dataset` with the resolved args.

- [ ] **Step 7.1: Write failing tests**

Create `libs/model-training/tests/test_reconstruction_cli.py`:

```python
"""Tests for model_training.reconstruction.cli."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_dry_run_emits_json_without_torch() -> None:
    from model_training.reconstruction import cli

    argv = [
        "--database-url", "sqlite:////tmp/fake.db",
        "--out-dir", "/tmp/recon_out",
        "--warm-start", "deltacoder",
        "--base-model", "qwen3.5-9b",
        "--min-fitness", "0.5",
        "--task-type", "bug-fix",
        "--emb-model", "none",
        "--no-zscore",
        "--dry-run",
    ]
    parser = cli._build_parser()
    args = parser.parse_args(argv)
    kwargs = cli._resolve_kwargs(args)
    assert kwargs["warm_start_adapter"] == "danielcherubini/Qwen3.5-DeltaCoder-9B"
    assert kwargs["base_model_id_override"] == "Qwen/Qwen3.5-9B"
    assert kwargs["min_fitness"] == 0.5
    assert kwargs["task_type"] == "bug-fix"
    assert kwargs["compute_zscore"] is False
    assert kwargs["emb_model_name"] is None
    assert kwargs["database_url"] == "sqlite:////tmp/fake.db"
    assert kwargs["out_dir"] == "/tmp/recon_out"


def test_dry_run_subprocess_does_not_import_torch(tmp_path: Path) -> None:
    # Run the CLI as a subprocess with torch masked out via PYTHONPATH trick:
    # we inspect the stdout JSON instead — success means no crash.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable, "-m", "model_training.reconstruction.cli",
            "--database-url", f"sqlite:///{tmp_path / 'fake.db'}",
            "--out-dir", str(tmp_path / "out"),
            "--warm-start", "off",
            "--base-model", "qwen3.5-9b",
            "--emb-model", "none",
            "--no-zscore",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["warm_start_adapter"] is None


def test_warm_start_aliases() -> None:
    from model_training.reconstruction.cli import _resolve_warm_start

    assert _resolve_warm_start("deltacoder") == "danielcherubini/Qwen3.5-DeltaCoder-9B"
    assert _resolve_warm_start("off") is None
    assert _resolve_warm_start("none") is None
    assert _resolve_warm_start("") is None
    assert _resolve_warm_start("custom/adapter") == "custom/adapter"


def test_base_model_aliases() -> None:
    from model_training.reconstruction.cli import _resolve_base_model

    assert _resolve_base_model("qwen3.5-9b") == "Qwen/Qwen3.5-9B"
    assert _resolve_base_model("Qwen/Qwen2.5-Coder-7B") == "Qwen/Qwen2.5-Coder-7B"


def test_cli_module_is_cpu_importable() -> None:
    import importlib

    mod = importlib.import_module("model_training.reconstruction.cli")
    assert hasattr(mod, "main")
```

- [ ] **Step 7.2: Run tests to verify they fail**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 7.3: Implement cli.py**

Create `libs/model-training/src/model_training/reconstruction/cli.py`:

```python
r"""CLI for building a reconstruction dataset.

Usage:
    uv run python -m model_training.reconstruction.cli \
        --database-url sqlite:///$HOME/.rune/rune.db \
        --out-dir data/recon_v1 \
        --warm-start deltacoder \
        --base-model qwen3.5-9b \
        --emb-model sentence-transformers/all-mpnet-base-v2

Mirrors ``trainer_cli.py`` conventions: ``--dry-run`` resolves args to JSON
and exits without importing torch; heavy imports are deferred.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WARM_START_ALIASES: dict[str, str | None] = {
    "deltacoder": "danielcherubini/Qwen3.5-DeltaCoder-9B",
    "off": None,
    "none": None,
    "": None,
}

_BASE_MODEL_ALIASES: dict[str, str] = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3-coder-next": "Qwen/Qwen3-Coder-Next",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recon_dataset_cli",
        description="Build a T2L reconstruction dataset from an AdapterRegistry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--warm-start",
        default="off",
        help="Warm-start adapter: 'deltacoder' alias, 'off'/'none' for none, or explicit id.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model alias or full HF repo id (overrides adapter_config.json).",
    )
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--min-fitness", type=float, default=None)
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source whitelist, e.g. 'distillation,evolution'.",
    )
    parser.add_argument(
        "--emb-model",
        default="none",
        help="HF repo id, 'default' for the built-in default, or 'none' for one-hot.",
    )
    parser.add_argument(
        "--no-zscore",
        dest="compute_zscore",
        action="store_false",
        default=True,
        help="Skip z-score stats computation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_warm_start(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in _WARM_START_ALIASES:
        return _WARM_START_ALIASES[key]
    return raw


def _resolve_base_model(raw: str) -> str:
    key = raw.strip().lower()
    if key in _BASE_MODEL_ALIASES:
        return _BASE_MODEL_ALIASES[key]
    return raw


def _resolve_sources(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _resolve_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    emb_choice = (args.emb_model or "none").strip().lower()
    if emb_choice in {"", "none"}:
        emb_model_name: str | None = None
    elif emb_choice == "default":
        from model_training.reconstruction.task_embeddings import (  # noqa: PLC0415
            DEFAULT_EMBEDDING_MODEL,
        )

        emb_model_name = DEFAULT_EMBEDDING_MODEL
    else:
        emb_model_name = args.emb_model
    return {
        "database_url": args.database_url,
        "out_dir": args.out_dir,
        "warm_start_adapter": _resolve_warm_start(args.warm_start),
        "base_model_id_override": _resolve_base_model(args.base_model),
        "task_type": args.task_type,
        "min_fitness": args.min_fitness,
        "sources": _resolve_sources(args.sources),
        "emb_model_name": emb_model_name,
        "compute_zscore": bool(args.compute_zscore),
    }


def _run(kwargs: dict[str, Any]) -> None:
    # Deferred imports so --dry-run stays torch-free.
    from sqlalchemy import create_engine  # noqa: PLC0415

    from adapter_registry.registry import AdapterRegistry  # noqa: PLC0415
    from model_training.reconstruction.builder import (  # noqa: PLC0415
        build_reconstruction_dataset,
    )
    from model_training.reconstruction.task_embeddings import (  # noqa: PLC0415
        DEFAULT_EMBEDDING_DIM,
        load_default_encoder,
    )

    engine = create_engine(kwargs["database_url"])
    registry = AdapterRegistry(engine=engine)

    emb_model_name = kwargs["emb_model_name"]
    emb_model: Any | None = None
    emb_dim = DEFAULT_EMBEDDING_DIM if emb_model_name else None
    if emb_model_name is not None:
        emb_model = load_default_encoder(emb_model_name)

    def _describe(rec: Any) -> str:
        # Default: prepend task_type to adapter id as a sanity-check description.
        # A real deployment should supply a richer callback by wrapping this CLI.
        return f"task_type={rec.task_type}; adapter_id={rec.id}"

    build_reconstruction_dataset(
        registry=registry,
        out_dir=Path(kwargs["out_dir"]),
        task_description_fn=_describe,
        warm_start_adapter=kwargs["warm_start_adapter"],
        base_model_id_override=kwargs["base_model_id_override"],
        emb_model=emb_model,
        compute_zscore=kwargs["compute_zscore"],
        task_type=kwargs["task_type"],
        min_fitness=kwargs["min_fitness"],
        sources=kwargs["sources"],
        emb_model_name=emb_model_name,
        emb_model_dim=emb_dim,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    kwargs = _resolve_kwargs(args)
    if args.dry_run:
        print(json.dumps(kwargs, indent=2, sort_keys=True))
        return 0
    _run(kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 7.4: Run tests to verify they pass**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_cli.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 7.5: Lint + type-check**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction/cli.py libs/model-training/tests/test_reconstruction_cli.py`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction/cli.py`
Expected: no errors.

- [ ] **Step 7.6: Commit**

```bash
git add libs/model-training/src/model_training/reconstruction/cli.py \
        libs/model-training/tests/test_reconstruction_cli.py
git commit -m "feat(reconstruction): CLI with dry-run + warm-start/base aliases"
```

---

## Task 8: End-to-End Integration Test

**Sequential. Requires Task 7 merged.**

**Files:**
- Modify: `libs/model-training/tests/test_reconstruction_builder.py` (append one E2E test)

**Acceptance:** A single pytest exercises the full pipeline: populate a registry with 3 adapters, run `build_reconstruction_dataset` with z-score enabled and one-hot embeddings, assert every manifest invariant, extract A/B from one of the adapter paths referenced by the manifest, verify the shape matches what a T2L trainer would consume — specifically assert `lora_A.shape == (n_layers, rank, in_features)` and `lora_B.shape == (n_layers, out_features, rank)`.

- [ ] **Step 8.1: Append the integration test**

Append to `libs/model-training/tests/test_reconstruction_builder.py` (at the bottom, before `test_builder_module_is_cpu_importable` if already defined, else at end):

```python
def test_e2e_manifest_points_at_adapters_that_extract_to_t2l_shape(
    tmp_path: Path, make_adapter_record: Callable[..., AdapterRecord]
) -> None:
    from model_training.reconstruction.builder import build_reconstruction_dataset
    from model_training.reconstruction.extract import (
        extract_lora_ab_from_state_dict,
        load_adapter_state_dict,
    )
    from model_training.reconstruction.manifest import ReconstructionManifest
    from model_training.reconstruction.stats import load_zscore_stats
    from model_training.reconstruction.task_embeddings import load_task_embeddings

    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    registry = AdapterRegistry(engine=engine)

    n_tasks = 3
    rank, in_features, out_features = 4, 16, 32
    layer_indices = [0, 1, 2]
    for i in range(n_tasks):
        adapter_dir = tmp_path / f"adapter-{i}"
        _write_fake_adapter(
            adapter_dir,
            base_model_name_or_path="danielcherubini/Qwen3.5-DeltaCoder-9B",
            rank=rank,
            target_modules=["q_proj", "v_proj"],
            layer_indices=layer_indices,
            in_features=in_features,
            out_features=out_features,
        )
        registry.store(
            make_adapter_record(
                id=f"adapter-{i}",
                rank=rank,
                file_path=str(adapter_dir),
                fitness_score=0.5 + 0.1 * i,
            )
        )

    out_dir = tmp_path / "recon_e2e"
    build_reconstruction_dataset(
        registry=registry,
        out_dir=out_dir,
        task_description_fn=lambda rec: f"task-{rec.id}",
        warm_start_adapter="danielcherubini/Qwen3.5-DeltaCoder-9B",
        base_model_id_override="Qwen/Qwen3.5-9B",
        emb_model=None,
        compute_zscore=True,
    )

    manifest = ReconstructionManifest.load(out_dir / "manifest.json")
    assert manifest.rank == rank
    assert len(manifest.records) == n_tasks

    embeddings = load_task_embeddings(out_dir / "task_embeddings.pt")
    assert set(embeddings) == {f"adapter-{i}" for i in range(n_tasks)}
    # One-hot over 3 tasks → 3x3 identity.
    stacked = torch.cat(
        [embeddings[f"adapter-{i}"] for i in range(n_tasks)], dim=0
    )
    assert torch.allclose(stacked @ stacked.T, torch.eye(n_tasks))

    stats = load_zscore_stats(out_dir / "zscore_stats.pt")
    assert set(stats) == {"q_proj", "v_proj"}
    # Stats tensors inherit the per-record tensor shapes.
    assert stats["q_proj"]["avg_A"].shape == (len(layer_indices), rank, in_features)
    assert stats["q_proj"]["std_A"].shape == (len(layer_indices), rank, in_features)
    assert stats["q_proj"]["avg_B"].shape == (len(layer_indices), out_features, rank)

    # Re-extract one oracle and confirm the T2L shape contract holds.
    first_adapter = Path(manifest.records[0].adapter_path)
    sd = load_adapter_state_dict(first_adapter)
    per_mod = extract_lora_ab_from_state_dict(sd, target_modules=manifest.target_modules)
    for module in manifest.target_modules:
        assert per_mod[module]["A"].shape == (len(layer_indices), rank, in_features)
        assert per_mod[module]["B"].shape == (len(layer_indices), out_features, rank)
        assert per_mod[module]["layer_indices"].tolist() == layer_indices
```

- [ ] **Step 8.2: Run only the new test to verify it passes**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_builder.py::test_e2e_manifest_points_at_adapters_that_extract_to_t2l_shape -v`
Expected: PASS.

- [ ] **Step 8.3: Run the full reconstruction test suite as a final gate**

Run: `uv run pytest libs/model-training/tests/test_reconstruction_manifest.py libs/model-training/tests/test_reconstruction_extract.py libs/model-training/tests/test_reconstruction_registry_source.py libs/model-training/tests/test_reconstruction_task_embeddings.py libs/model-training/tests/test_reconstruction_stats.py libs/model-training/tests/test_reconstruction_builder.py libs/model-training/tests/test_reconstruction_cli.py -v`
Expected: ALL PASS.

- [ ] **Step 8.4: Run ruff + mypy on the full subpackage**

Run: `uv run ruff check libs/model-training/src/model_training/reconstruction`
Run: `uv run mypy libs/model-training/src/model_training/reconstruction`
Expected: no errors.

- [ ] **Step 8.5: Commit**

```bash
git add libs/model-training/tests/test_reconstruction_builder.py
git commit -m "test(reconstruction): end-to-end shape + embedding + stats invariants"
```

---

## Out of Scope (Explicit Non-Goals)

- **Hypernetwork architecture swap.** That is a separate P0 in the fit-assessment YAML. This plan only produces its *training dataset*.
- **Corpus production runs.** No HPO-driven `(task, step)` sweep — this plan consumes an already-populated registry.
- **Pass@1 kill-switch gate.** P1 followup; not wired here.
- **Delta_w_scaling audit** (0.16 vs T2L 10000). Deferred per YAML; our builder emits raw A/B, leaving the scaling decision to the hypernetwork trainer.
- **`record_trajectory` dead-code fix.** Unrelated P2; the registry already has usable adapters without it.

## Downstream Consumer Contract

A future hypernetwork trainer that consumes `manifest.json` should:

1. `ReconstructionManifest.load(manifest_path)`.
2. `load_task_embeddings(manifest_path.parent / "task_embeddings.pt")`.
3. Optionally `load_zscore_stats(Path(manifest.zscore_stats_path))`.
4. For each `record` in `manifest.records`, at training time:
   - `sd = load_adapter_state_dict(Path(record.adapter_path))`
   - `per_mod = extract_lora_ab_from_state_dict(sd, target_modules=manifest.target_modules)`
   - Batch per-module A/B tensors across tasks for the reconstruction L1 loss.
5. MUST load base + `manifest.warm_start_adapter` at inference, in that order, before applying the generated delta.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-reconstruction-dataset-builder.md`. Two execution options:

1. **Subagent-Driven (recommended).** Dispatch a fresh subagent per task. Tasks 2–5 run in parallel after Task 1 lands. Tasks 6–8 run sequentially.
2. **Inline Execution.** Single session runs tasks 1 → 8 in order.

Which approach?
