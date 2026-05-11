"""Trajectory-aware RAG pipeline for Condition (ii) baseline.

Builds a FAISS vector store from mined trajectory corpus, retrieves
relevant trajectory chunks at inference time using (state, goal) queries.

GPU-heavy imports deferred per INFRA-05.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG pipeline.

    Attributes:
        embedding_model: sentence-transformers model id.
        chunk_size: Token count per chunk (approximate, whitespace-split).
        chunk_overlap: Overlap between consecutive chunks in tokens.
        top_k: Number of chunks to retrieve per query.
        reranker: Optional cross-encoder reranker model id.
    """

    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    reranker: str | None = None


def _get_encoder(model_id: str) -> Any:
    """Load sentence-transformers encoder (deferred import)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by approximate token count.

    Args:
        text: Source text.
        chunk_size: Target tokens per chunk (~4 chars/token).
        overlap: Overlap in tokens.

    Returns:
        List of chunk strings.
    """
    char_chunk = chunk_size * 4
    char_overlap = overlap * 4
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_chunk
        chunks.append(text[start:end])
        start = end - char_overlap
    return chunks


def build_vector_store(
    corpus_path: Path,
    config: RAGConfig,
) -> dict[str, Any]:
    """Build a FAISS index from a trajectory corpus.

    Args:
        corpus_path: JSONL with "trajectory" field per line.
        config: RAG configuration.

    Returns:
        Dict with "index" (faiss.IndexFlatIP), "chunks" (list[str]),
        "n_chunks" (int).
    """
    import faiss
    import numpy as np

    encoder = _get_encoder(config.embedding_model)

    all_chunks: list[str] = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            traj = record.get("trajectory", "")
            chunks = _chunk_text(traj, config.chunk_size, config.chunk_overlap)
            all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"No chunks produced from {corpus_path}")

    embeddings = encoder.encode(
        all_chunks, convert_to_numpy=True, show_progress_bar=True
    )
    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info("Built FAISS index: %d chunks, dim=%d", len(all_chunks), dim)

    return {"index": index, "chunks": all_chunks, "n_chunks": len(all_chunks)}


def query_trajectory_rag(
    query: str,
    index: Any,
    chunks: list[str],
    encoder: Any,
    top_k: int = 5,
) -> list[str]:
    """Retrieve top-k trajectory chunks for a query.

    Args:
        query: Natural language query (current state + goal).
        index: FAISS index.
        chunks: List of chunk strings aligned with index vectors.
        encoder: Sentence-transformers encoder with .encode().
        top_k: Number of results.

    Returns:
        List of retrieved chunk strings, ranked by relevance.
    """
    import numpy as np

    q_emb = encoder.encode([query], convert_to_numpy=True).astype(np.float32)

    import faiss

    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)
    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results
