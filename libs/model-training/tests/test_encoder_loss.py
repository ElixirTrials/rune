"""Tests for InfoNCE contrastive loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_infonce_loss_is_scalar() -> None:
    from model_training.encoder_pretrain.loss import infonce_loss

    # embeddings: batch_size=4, dim=8
    anchors = torch.randn(4, 8)
    positives = torch.randn(4, 8)
    loss_val = infonce_loss(anchors, positives, temperature=0.07)
    assert loss_val.shape == ()  # scalar
    assert loss_val.item() > 0.0


def test_infonce_loss_lower_for_matched_pairs() -> None:
    """Loss is lower when anchors and positives are identical (trivial case)."""
    from model_training.encoder_pretrain.loss import infonce_loss

    embeddings = torch.randn(8, 16)
    loss_matched = infonce_loss(embeddings, embeddings.clone(), temperature=0.07)
    loss_random = infonce_loss(embeddings, torch.randn(8, 16), temperature=0.07)
    # matched embeddings should have lower loss than random negatives
    assert loss_matched.item() < loss_random.item()


def test_infonce_loss_has_gradient() -> None:
    """Loss is differentiable with respect to anchor embeddings."""
    from model_training.encoder_pretrain.loss import infonce_loss

    anchors = torch.randn(4, 8, requires_grad=True)
    positives = torch.randn(4, 8)
    loss_val = infonce_loss(anchors, positives, temperature=0.07)
    loss_val.backward()
    assert anchors.grad is not None
    assert not torch.isnan(anchors.grad).any()


def test_infonce_loss_temperature_scaling() -> None:
    """Lower temperature sharpens the distribution (different loss for random pairs)."""
    from model_training.encoder_pretrain.loss import infonce_loss

    anchors = torch.randn(4, 8)
    positives = torch.randn(4, 8)
    loss_sharp = infonce_loss(anchors, positives, temperature=0.01)
    loss_flat = infonce_loss(anchors, positives, temperature=1.0)
    # Sharper temperature produces different (usually higher) loss for mismatched pairs
    assert loss_sharp.item() != loss_flat.item()


def test_infonce_loss_shape_mismatch_raises() -> None:
    """Mismatched anchor/positive shapes raise ValueError."""
    from model_training.encoder_pretrain.loss import infonce_loss

    anchors = torch.randn(4, 8)
    positives = torch.randn(4, 16)
    with pytest.raises(ValueError, match="shape"):
        infonce_loss(anchors, positives)
