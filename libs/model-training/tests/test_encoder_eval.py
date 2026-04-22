"""Tests for retrieval eval (MRR@10, Recall@1) with synthetic data.

All tests use mocked/synthetic encoders — no GPU or network required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")


def _make_identity_model(dim: int = 16) -> Any:
    """Return a mock model whose forward pass embeds input_ids as identity rows.

    For batch size B and sequence length S, returns last_hidden_state of
    shape (B, S, dim) where each token position is a scaled one-hot row.
    This ensures mean-pooled embeddings are deterministic per-row index.
    """
    import torch

    class _Output:
        def __init__(self, t: torch.Tensor) -> None:
            self.last_hidden_state = t

    class _FakeModel:
        def __call__(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> _Output:
            B, S = input_ids.shape
            # Each row i gets embedding e_i (unit vector in dim i % dim direction)
            # by broadcasting a (B, dim) tensor across the sequence length
            base = torch.zeros(B, dim)
            for i in range(B):
                base[i, i % dim] = 1.0
            # Expand to (B, S, dim)
            hidden = base.unsqueeze(1).expand(B, S, dim)
            return _Output(hidden)

        def eval(self) -> "_FakeModel":
            return self

        def train(self) -> "_FakeModel":
            return self

    return _FakeModel()


def _make_fake_tokenizer(max_length: int = 32) -> Any:
    """Return a mock tokenizer that returns fixed-shape tensors."""
    import torch

    class _FakeTok:
        def __call__(
            self,
            texts: list[str],
            padding: str = "max_length",
            truncation: bool = True,
            max_length: int = max_length,
            return_tensors: str = "pt",
        ) -> dict[str, torch.Tensor]:
            B = len(texts)
            return {
                "input_ids": torch.zeros(B, max_length, dtype=torch.long),
                "attention_mask": torch.ones(B, max_length, dtype=torch.long),
            }

    return _FakeTok()


def _make_test_rows(n: int = 10) -> list[dict[str, Any]]:
    """Build synthetic test rows with identical encoder_input and post_code."""
    return [
        {
            "task_id": f"pr_{i:03d}",
            "encoder_input": f"task {i}",
            "post_code": f"task {i}",
            "pre_code": "",
            "task_desc": f"task {i}",
            "task_desc_source": "explicit_field",
            "metadata": {},
        }
        for i in range(n)
    ]


def test_run_retrieval_eval_returns_expected_keys() -> None:
    """run_retrieval_eval returns dict with mrr_at_10 and recall_at_1."""
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval

    model = _make_identity_model(dim=16)
    tokenizer = _make_fake_tokenizer()
    rows = _make_test_rows(n=5)

    metrics = run_retrieval_eval(
        model=model,
        tokenizer=tokenizer,
        test_rows=rows,
        max_length=32,
        batch_size=5,
        device="cpu",
    )

    assert "mrr_at_10" in metrics
    assert "recall_at_1" in metrics
    assert 0.0 <= metrics["mrr_at_10"] <= 1.0
    assert 0.0 <= metrics["recall_at_1"] <= 1.0


def test_run_retrieval_eval_returns_zero_for_empty() -> None:
    """Empty test_rows returns zeros without error."""
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval

    model = _make_identity_model()
    tokenizer = _make_fake_tokenizer()

    metrics = run_retrieval_eval(
        model=model,
        tokenizer=tokenizer,
        test_rows=[],
        max_length=32,
        batch_size=10,
        device="cpu",
    )
    assert metrics == {"mrr_at_10": 0.0, "recall_at_1": 0.0}


def test_run_retrieval_eval_perfect_recall_for_identical_pairs() -> None:
    """When anchor and positive are identical texts, recall@1 should be 1.0."""
    import torch
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval

    # Use a model that returns the same embedding for any input (degenerate, but
    # that means all anchors == all positives so rank is always 1 due to tie-breaking
    # at position 0 — or at minimum recall@1 >= some threshold).
    # Instead, use a model that differentiates rows by batch position.
    model = _make_identity_model(dim=64)
    tokenizer = _make_fake_tokenizer()

    # With 4 rows and identity model: each row i gets embedding e_{i%64}
    # So anchor[i] == positive[i] for all i, meaning diagonal similarity is highest.
    rows = _make_test_rows(n=4)
    metrics = run_retrieval_eval(
        model=model,
        tokenizer=tokenizer,
        test_rows=rows,
        max_length=32,
        batch_size=4,
        device="cpu",
    )
    # With orthogonal identity embeddings, each anchor matches exactly its positive
    assert metrics["recall_at_1"] == 1.0
    assert metrics["mrr_at_10"] == 1.0


def test_run_retrieval_eval_float_values() -> None:
    """Metrics are Python floats, not tensors."""
    from model_training.encoder_pretrain.eval_encoder import run_retrieval_eval

    model = _make_identity_model()
    tokenizer = _make_fake_tokenizer()
    rows = _make_test_rows(n=3)

    metrics = run_retrieval_eval(
        model=model,
        tokenizer=tokenizer,
        test_rows=rows,
        max_length=32,
        batch_size=3,
        device="cpu",
    )
    assert isinstance(metrics["mrr_at_10"], float)
    assert isinstance(metrics["recall_at_1"], float)
