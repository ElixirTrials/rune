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
