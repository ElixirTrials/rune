"""Tests for trajectory-aware RAG pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from model_training.rag_pipeline import (
    RAGConfig,
    build_vector_store,
    query_trajectory_rag,
)


def test_rag_config_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.chunk_size > 0
    assert cfg.top_k > 0
    assert cfg.embedding_model is not None


def test_build_vector_store_returns_index(tmp_path) -> None:
    """build_vector_store creates a FAISS index from trajectory chunks."""
    import json

    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"trajectory": f"def task_{i}():\n    return {i}\n", "task_id": f"t{i}"}
        for i in range(5)
    ]
    with corpus.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with patch("model_training.rag_pipeline._get_encoder") as mock_enc:
        import numpy as np

        mock_enc.return_value.encode.return_value = np.random.randn(5, 768).astype(
            np.float32
        )
        store = build_vector_store(corpus, RAGConfig(chunk_size=512))

    assert store["n_chunks"] >= 5


def test_query_returns_top_k() -> None:
    """query_trajectory_rag returns at most top_k results."""
    import numpy as np

    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.9, 0.8, 0.7]]),
        np.array([[0, 1, 2]]),
    )
    chunks = ["chunk0", "chunk1", "chunk2", "chunk3"]

    results = query_trajectory_rag(
        query="def foo():",
        index=mock_index,
        chunks=chunks,
        encoder=MagicMock(
            encode=MagicMock(return_value=np.random.randn(1, 768).astype(np.float32))
        ),
        top_k=3,
    )
    assert len(results) == 3
