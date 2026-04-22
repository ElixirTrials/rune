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
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]  # noqa: PLC0415, I001

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

    Args:
        descriptions: Mapping of task_id to natural-language description.
        model: Encoder with an ``encode`` method, or ``None`` for one-hot.

    Returns:
        Dict mapping task_id to a ``Tensor[1, dim]``.
    """
    import torch  # noqa: PLC0415

    task_ids = list(descriptions.keys())
    n = len(task_ids)

    if model is None:
        logger.info(
            "compute_task_embeddings: no encoder → one-hot fallback (dim=%d)", n
        )
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
    """Persist ``{task_id: Tensor}`` as a ``torch.save`` file.

    Args:
        embeddings: Dict mapping task_id to embedding tensor.
        path: Destination file path (parent dirs created if needed).
    """
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, str(path))


def load_task_embeddings(path: Path) -> dict[str, Any]:
    """Inverse of ``save_task_embeddings``.

    Args:
        path: Path to a file saved by ``save_task_embeddings``.

    Returns:
        Dict mapping task_id to embedding tensor.

    Raises:
        ValueError: If the loaded object is not a dict.
    """
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
