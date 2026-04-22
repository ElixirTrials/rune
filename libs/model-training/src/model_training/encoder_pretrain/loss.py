"""InfoNCE contrastive loss for encoder pretraining.

Uses in-batch negatives: for a batch of size N, the positive pair for
anchor i is positive i; all other j != i are treated as negatives.

All imports are deferred inside function bodies (INFRA-05).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def infonce_loss(
    anchors: Any,
    positives: Any,
    temperature: float = 0.07,
) -> Any:
    """Compute InfoNCE (NT-Xent) loss with in-batch negatives.

    Both tensors must be embeddings of shape ``(batch_size, embedding_dim)``.
    The loss is symmetric: mean of (anchor->positive CE) and (positive->anchor CE).
    Embeddings are L2-normalised internally before computing similarities.

    Args:
        anchors: Anchor embeddings ``(B, D)`` — a ``torch.Tensor``.
        positives: Positive embeddings ``(B, D)``. Must be same shape as anchors.
        temperature: Temperature scalar tau. Lower = harder negatives.

    Returns:
        Scalar ``torch.Tensor`` loss with gradient enabled.

    Raises:
        ValueError: If anchors and positives have different shapes.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415, N812

    if anchors.shape != positives.shape:
        raise ValueError(
            f"anchors shape {anchors.shape} != positives shape {positives.shape}"
        )

    batch_size = anchors.shape[0]

    # L2-normalise both sides
    anchors_norm = F.normalize(anchors, dim=-1)
    positives_norm = F.normalize(positives, dim=-1)

    # Similarity matrix: (B, B), scaled by temperature
    logits = torch.matmul(anchors_norm, positives_norm.T) / temperature

    # Diagonal entries are the positive pairs
    labels = torch.arange(batch_size, device=anchors.device)

    # Symmetric loss: anchor -> positive and positive -> anchor
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)

    return (loss_a + loss_b) / 2.0
