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
