"""Tests for model_training.diff_loss — CPU-only (no torch/trl at collection time).

Legacy set-based path tests are preserved as-is.  New hunk-path tests are
added below.  All tests must pass without GPU libraries installed.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100


def _make_mock_tokenizer(
    post_code: str,
    token_ids: list[int],
    offsets: list[tuple[int, int]],
) -> Any:
    """Return a MagicMock tokenizer whose __call__ returns known ids + offsets."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": token_ids,
        "offset_mapping": offsets,
    }
    tok.decode = MagicMock(side_effect=lambda ids, **kw: post_code[: len(ids)])
    return tok


def _make_inner_collator(
    input_ids: list[list[int]],
    labels: list[list[int]],
) -> Any:
    """Return a fake inner collator that returns tensors built from the given lists."""
    import torch

    def _collate(features: list[dict]) -> dict:
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
        }

    return _collate


# ---------------------------------------------------------------------------
# Legacy set-based path tests
# ---------------------------------------------------------------------------


class TestComputeDiffLossWeights:
    def test_masked_positions_get_zero(self) -> None:
        from model_training.diff_loss import compute_diff_loss_weights

        ids = [1, 2, 3]
        labels = [IGNORE_INDEX, 2, IGNORE_INDEX]
        w = compute_diff_loss_weights(ids, labels, changed_ids={1, 2, 3})
        assert w == [0.0, 1.0, 0.0]

    def test_changed_token_gets_changed_weight(self) -> None:
        from model_training.diff_loss import compute_diff_loss_weights

        ids = [10, 20, 30]
        labels = [10, 20, 30]
        w = compute_diff_loss_weights(
            ids, labels, changed_ids={20}, changed_weight=2.0, unchanged_weight=0.5
        )
        assert w == [0.5, 2.0, 0.5]

    def test_empty_changed_ids_all_unchanged(self) -> None:
        from model_training.diff_loss import compute_diff_loss_weights

        ids = [1, 2, 3]
        labels = [1, 2, 3]
        w = compute_diff_loss_weights(ids, labels, changed_ids=set())
        assert all(x == 0.3 for x in w)

    def test_identity_under_uniform_weights(self) -> None:
        """Legacy identity invariant: uniform weights → every non-masked token == w."""
        from model_training.diff_loss import compute_diff_loss_weights

        ids = [1, 2, 3, 4]
        labels = [IGNORE_INDEX, 2, 3, IGNORE_INDEX]
        w = compute_diff_loss_weights(
            ids,
            labels,
            changed_ids={1, 2, 3, 4},
            changed_weight=1.0,
            unchanged_weight=1.0,
        )
        assert w == [0.0, 1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# _compute_hunk_ranges tests
# ---------------------------------------------------------------------------


class TestComputeHunkRanges:
    def test_equal_strings_returns_empty(self) -> None:
        from model_training.diff_loss import _compute_hunk_ranges

        assert _compute_hunk_ranges("abc\n", "abc\n") == []

    def test_basic_replacement(self) -> None:
        """before='a\\nb\\nc\\n', after='a\\nB\\nc\\n' → one hunk for line 2."""
        from model_training.diff_loss import _compute_hunk_ranges

        before = "a\nb\nc\n"
        after = "a\nB\nc\n"
        ranges = _compute_hunk_ranges(before, after)
        assert len(ranges) == 1
        # Line 2 in after starts at char 2 ("a\n" = 2 chars) and is "B\n" = 2 chars.
        assert ranges[0] == (2, 4)
        # Verify the slice is correct.
        assert after[ranges[0][0] : ranges[0][1]] == "B\n"

    def test_pure_insertion(self) -> None:
        """before='a\\nc\\n', after='a\\nb\\nc\\n' → hunk covering inserted line."""
        from model_training.diff_loss import _compute_hunk_ranges

        before = "a\nc\n"
        after = "a\nb\nc\n"
        ranges = _compute_hunk_ranges(before, after)
        assert len(ranges) == 1
        start, end = ranges[0]
        assert after[start:end] == "b\n"

    def test_pure_deletion(self) -> None:
        """before='a\\nb\\nc\\n', after='a\\nc\\n' → no hunk in after."""
        from model_training.diff_loss import _compute_hunk_ranges

        before = "a\nb\nc\n"
        after = "a\nc\n"
        ranges = _compute_hunk_ranges(before, after)
        assert ranges == []

    def test_modification(self) -> None:
        """before='x = 1\\n', after='x = 2\\n' → one hunk covering the full line."""
        from model_training.diff_loss import _compute_hunk_ranges

        before = "x = 1\n"
        after = "x = 2\n"
        ranges = _compute_hunk_ranges(before, after)
        assert len(ranges) == 1
        assert after[ranges[0][0] : ranges[0][1]] == "x = 2\n"

    def test_multi_hunk(self) -> None:
        """Two separate modification blocks produce two disjoint ranges."""
        from model_training.diff_loss import _compute_hunk_ranges

        before = "a\nb\nc\nd\ne\n"
        after = "a\nB\nc\nD\ne\n"
        ranges = _compute_hunk_ranges(before, after)
        assert len(ranges) == 2
        # Ranges must be disjoint and ascending.
        assert ranges[0][1] <= ranges[1][0]
        assert after[ranges[0][0] : ranges[0][1]] == "B\n"
        assert after[ranges[1][0] : ranges[1][1]] == "D\n"


# ---------------------------------------------------------------------------
# compute_hunk_loss_weights tests
# ---------------------------------------------------------------------------


class TestComputeHunkLossWeights:
    def test_masked_positions_get_zero(self) -> None:
        from model_training.diff_loss import compute_hunk_loss_weights

        input_ids = [1, 2, 3]
        labels = [IGNORE_INDEX, 2, IGNORE_INDEX]
        offsets = [(0, 1), (1, 2), (2, 3)]
        hunk_ranges = [(1, 2)]  # token 1 (id=2) is in hunk
        w = compute_hunk_loss_weights(input_ids, labels, offsets, hunk_ranges)
        assert w[0] == 0.0
        assert w[2] == 0.0

    def test_special_tokens_get_zero(self) -> None:
        """offset_mapping (0,0) → special token → weight 0 regardless of label."""
        from model_training.diff_loss import compute_hunk_loss_weights

        input_ids = [0, 10, 20]
        labels = [1, 10, 20]  # all non-masked
        offsets = [(0, 0), (0, 3), (3, 6)]
        hunk_ranges = [(0, 6)]  # everything in hunk
        w = compute_hunk_loss_weights(input_ids, labels, offsets, hunk_ranges)
        assert w[0] == 0.0  # special token despite non-masked label

    def test_tokens_in_range_get_changed_weight(self) -> None:
        from model_training.diff_loss import compute_hunk_loss_weights

        input_ids = [1, 2, 3, 4]
        labels = [1, 2, 3, 4]
        offsets = [(0, 2), (2, 4), (4, 6), (6, 8)]
        hunk_ranges = [(2, 6)]  # tokens 1 and 2 (0-indexed) are in hunk
        w = compute_hunk_loss_weights(
            input_ids,
            labels,
            offsets,
            hunk_ranges,
            changed_weight=2.0,
            unchanged_weight=0.1,
        )
        assert w[0] == pytest.approx(0.1)  # outside hunk
        assert w[1] == pytest.approx(2.0)  # in hunk [2,6), offset (2,4): 2<6 and 4>2
        assert w[2] == pytest.approx(2.0)  # in hunk [2,6), offset (4,6): 4<6 and 6>2
        assert w[3] == pytest.approx(0.1)  # outside hunk

    def test_identity_under_uniform_weights_hunk_path(self) -> None:
        """CRITICAL: changed==unchanged → all non-masked non-special tokens == w."""
        from model_training.diff_loss import compute_hunk_loss_weights

        input_ids = [1, 2, 3, 4, 5]
        labels = [IGNORE_INDEX, 2, 3, 4, IGNORE_INDEX]
        offsets = [(0, 0), (0, 3), (3, 6), (6, 9), (0, 0)]
        hunk_ranges = [(0, 5)]  # partial hunk — doesn't matter for identity
        w = compute_hunk_loss_weights(
            input_ids,
            labels,
            offsets,
            hunk_ranges,
            changed_weight=1.0,
            unchanged_weight=1.0,
        )
        # Masked positions → 0
        assert w[0] == 0.0
        assert w[4] == 0.0
        # Non-masked, non-special → 1.0 regardless of hunk membership
        assert w[1] == pytest.approx(1.0)
        assert w[2] == pytest.approx(1.0)
        assert w[3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DiffWeightedDataCollator tests
# ---------------------------------------------------------------------------


class TestDiffWeightedDataCollator:
    def _make_features(
        self,
        n: int = 1,
        include_pre_post: bool = True,
        pre_code: str = "x = 1\n",
        post_code: str = "x = 2\n",
    ) -> list[dict]:
        feats = []
        for _ in range(n):
            f: dict = {"text": "hello"}
            if include_pre_post:
                f["pre_code"] = pre_code
                f["post_code"] = post_code
            feats.append(f)
        return feats

    def test_collator_prefers_hunk_path_when_pre_post_present(self) -> None:
        """When features have pre/post AND tokenizer is set, hunk path is used."""
        import torch  # noqa: F401
        from model_training.diff_loss import DiffWeightedDataCollator

        input_ids = [[1, 2, 3]]
        labels = [[IGNORE_INDEX, 2, 3]]
        inner = _make_inner_collator(input_ids, labels)

        post_code = "x = 2\n"
        tok = _make_mock_tokenizer(
            post_code,
            token_ids=[2, 3],
            offsets=[(0, 3), (3, 6)],
        )

        collator = DiffWeightedDataCollator(inner, tokenizer=tok)
        features = self._make_features(n=1, pre_code="x = 1\n", post_code=post_code)

        with (
            patch(
                "model_training.diff_loss.compute_hunk_loss_weights",
                wraps=__import__(
                    "model_training.diff_loss", fromlist=["compute_hunk_loss_weights"]
                ).compute_hunk_loss_weights,
            ) as mock_hunk,
            patch(
                "model_training.diff_loss.compute_diff_loss_weights",
                wraps=__import__(
                    "model_training.diff_loss", fromlist=["compute_diff_loss_weights"]
                ).compute_diff_loss_weights,
            ) as mock_legacy,
        ):
            batch = collator(features)

        assert mock_hunk.called, "hunk path should have been used"
        assert not mock_legacy.called, "legacy path should NOT have been used"
        assert "loss_weights" in batch
        assert batch["loss_weights"].shape == torch.Size([1, 3])

    def test_collator_falls_back_on_missing_pre_post(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When features lack pre/post, uses identity weights and warns."""
        import model_training.diff_loss as dl_module
        import torch
        from model_training.diff_loss import DiffWeightedDataCollator

        # Reset global flag so warning fires.
        dl_module._HUNK_FALLBACK_WARNED = False

        input_ids = [[1, 2, 3]]
        labels = [[IGNORE_INDEX, 2, 3]]
        inner = _make_inner_collator(input_ids, labels)
        tok = MagicMock()
        collator = DiffWeightedDataCollator(inner, tokenizer=tok)

        features = self._make_features(n=1, include_pre_post=False)

        with caplog.at_level(logging.WARNING, logger="model_training.diff_loss"):
            batch = collator(features)

        # Identity fallback: IGNORE_INDEX (-100) → 0.0, labeled tokens → 1.0.
        assert "loss_weights" in batch
        assert batch["loss_weights"].shape == torch.Size([1, 3])
        weights = batch["loss_weights"][0].tolist()
        assert weights[0] == 0.0, "IGNORE_INDEX position should get 0.0"
        assert weights[1] == 1.0, "labeled token should get 1.0"
        assert weights[2] == 1.0, "labeled token should get 1.0"
        assert any(
            "pre_code/post_code" in rec.message or "fallback" in rec.message.lower()
            for rec in caplog.records
        ), "expected a fallback warning in logs"


# ---------------------------------------------------------------------------
# _compute_weighted_loss tests (pure function, no Trainer needed)
# ---------------------------------------------------------------------------


class TestComputeWeightedLoss:
    """Unit tests for the pure _compute_weighted_loss helper.

    All tests use a tiny stub model that returns fixed logits; no trl/GPU
    required.
    """

    def _make_inputs(
        self,
        batch: int = 1,
        seq: int = 4,
        vocab: int = 8,
        seed: int = 0,
    ):
        """Return (logits, labels, all-ones weights) tensors."""
        import torch

        torch.manual_seed(seed)
        logits = torch.randn(batch, seq, vocab)
        # Labels: first and last positions masked, middle ones active.
        labels = torch.full((batch, seq), IGNORE_INDEX, dtype=torch.long)
        for b in range(batch):
            for s in range(1, seq - 1):
                labels[b, s] = (b * seq + s) % vocab
        loss_weights = torch.ones(batch, seq)
        return logits, labels, loss_weights

    def test_identity_under_uniform_weights(self) -> None:
        """All weights == 1.0 → matches standard mean CE on labeled tokens."""
        import torch.nn.functional as F  # noqa: N812
        from model_training.diff_loss import _compute_weighted_loss

        logits, labels, loss_weights = self._make_inputs(batch=1, seq=4)

        got = _compute_weighted_loss(logits, labels, loss_weights)

        # Reference: standard CE with shift, mean over non-masked shifted tokens.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ref = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )

        assert got.item() == pytest.approx(ref.item(), rel=1e-5)

    def test_weighted_loss_correctness(self) -> None:
        """Manually computed weighted mean matches _compute_weighted_loss."""
        import torch
        import torch.nn.functional as F  # noqa: N812
        from model_training.diff_loss import _compute_weighted_loss

        logits, labels, _ = self._make_inputs(batch=1, seq=4)
        # weights: [0.5, 1.0, 0.0, 2.0]
        loss_weights = torch.tensor([[0.5, 1.0, 0.0, 2.0]])

        got = _compute_weighted_loss(logits, labels, loss_weights)

        # Manual reference (causal shift: predict token[1..] from logits[0..]).
        shift_logits = logits[:, :-1, :]  # [1, 3, V]
        shift_labels = labels[:, 1:]  # [1, 3]
        shift_weights = loss_weights[:, 1:]  # [1, 3] → [1.0, 0.0, 2.0]

        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(shift_labels.shape)  # [1, 3]

        label_mask = (shift_labels != IGNORE_INDEX).float()
        numerator = (per_token * shift_weights * label_mask).sum()
        denominator = (shift_weights * label_mask).sum()
        expected = (numerator / denominator).item()

        assert got.item() == pytest.approx(expected, rel=1e-5)

    def test_loss_weights_popped_from_inputs(self) -> None:
        """loss_weights must NOT reach model(**inputs)."""
        import torch
        from model_training.diff_loss import DiffAwareSFTTrainer

        # Build a minimal trainer subclass that overrides __init__ to avoid
        # the heavy SFTTrainer constructor.
        class _StubTrainer(DiffAwareSFTTrainer):
            def __init__(self) -> None:  # type: ignore[override]
                pass  # skip SFTTrainer.__init__

        trainer = _StubTrainer()

        logits_tensor = torch.randn(1, 4, 8)
        labels_tensor = torch.tensor([[IGNORE_INDEX, 1, 2, IGNORE_INDEX]])

        received_keys: list[list[str]] = []

        class _FakeOutputs:
            logits = logits_tensor

        def _fake_model(**kwargs):  # type: ignore[return]
            received_keys.append(list(kwargs.keys()))
            return _FakeOutputs()

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": labels_tensor,
            "loss_weights": torch.ones(1, 4),
        }
        trainer.compute_loss(_fake_model, inputs, return_outputs=False)  # type: ignore[arg-type]

        assert received_keys, "model was never called"
        assert "loss_weights" not in received_keys[0], (
            "loss_weights leaked into model(**inputs)"
        )

    def test_missing_loss_weights_no_keyerror(self) -> None:
        """When batch has no loss_weights, compute_loss does not raise."""
        import torch
        from model_training.diff_loss import DiffAwareSFTTrainer

        class _StubTrainer(DiffAwareSFTTrainer):
            def __init__(self) -> None:  # type: ignore[override]
                pass

        trainer = _StubTrainer()

        logits_tensor = torch.randn(1, 4, 8)
        labels_tensor = torch.tensor([[IGNORE_INDEX, 1, 2, IGNORE_INDEX]])

        class _FakeOutputs:
            logits = logits_tensor
            loss = torch.tensor(1.23)

        def _fake_model(**kwargs):  # type: ignore[return]
            return _FakeOutputs()

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": labels_tensor,
            # no "loss_weights" key
        }

        # Should not raise; falls back to outputs.loss.
        loss = trainer.compute_loss(_fake_model, inputs, return_outputs=False)  # type: ignore[arg-type]
        assert loss is not None

    def test_batch_shape_handling(self) -> None:
        """batch=2, seq=3: weighted reduction is correct across rows."""
        import torch
        import torch.nn.functional as F  # noqa: N812
        from model_training.diff_loss import _compute_weighted_loss

        torch.manual_seed(42)
        batch, seq, vocab = 2, 3, 6
        logits = torch.randn(batch, seq, vocab)
        labels = torch.tensor(
            [
                [IGNORE_INDEX, 2, 3],
                [IGNORE_INDEX, 4, 1],
            ]
        )
        loss_weights = torch.tensor(
            [
                [1.0, 2.0, 0.5],
                [1.0, 0.5, 3.0],
            ]
        )

        got = _compute_weighted_loss(logits, labels, loss_weights)

        # Manual reference.
        shift_logits = logits[:, :-1, :]  # [2, 2, V]
        shift_labels = labels[:, 1:]  # [2, 2]
        shift_weights = loss_weights[:, 1:]  # [2, 2]
        per_token = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(batch, seq - 1)
        label_mask = (shift_labels != IGNORE_INDEX).float()
        expected = (
            (per_token * shift_weights * label_mask).sum()
            / (shift_weights * label_mask).sum()
        ).item()

        assert got.item() == pytest.approx(expected, rel=1e-5)

    def test_return_outputs_tuple(self) -> None:
        """return_outputs=True returns (loss, outputs) tuple."""
        import torch
        from model_training.diff_loss import DiffAwareSFTTrainer

        class _StubTrainer(DiffAwareSFTTrainer):
            def __init__(self) -> None:  # type: ignore[override]
                pass

        trainer = _StubTrainer()

        logits_tensor = torch.randn(1, 4, 8)
        labels_tensor = torch.tensor([[IGNORE_INDEX, 1, 2, IGNORE_INDEX]])

        class _FakeOutputs:
            logits = logits_tensor

        def _fake_model(**kwargs):  # type: ignore[return]
            return _FakeOutputs()

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": labels_tensor,
            "loss_weights": torch.ones(1, 4),
        }
        result = trainer.compute_loss(_fake_model, inputs, return_outputs=True)  # type: ignore[arg-type]
        assert isinstance(result, tuple) and len(result) == 2
        loss, outputs = result
        assert hasattr(loss, "item"), "first element should be a scalar tensor"
        assert isinstance(outputs, _FakeOutputs)


# ---------------------------------------------------------------------------
# Integration test: factory returns DiffAwareSFTTrainer (requires trl)
# ---------------------------------------------------------------------------


class TestBuildDiffAwareSftTrainerIntegration:
    def test_factory_returns_diff_aware_trainer(self) -> None:
        """build_diff_aware_sft_trainer returns a DiffAwareSFTTrainer instance."""
        pytest.importorskip("trl", reason="trl not installed")

        from unittest.mock import MagicMock

        from model_training.diff_loss import (
            DiffAwareSFTTrainer,
            build_diff_aware_sft_trainer,
        )

        model = MagicMock()
        args = MagicMock()
        dataset = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        trainer = build_diff_aware_sft_trainer(
            model,
            args,
            dataset,
            processing_class=tokenizer,
        )

        assert isinstance(trainer, DiffAwareSFTTrainer), (
            f"Expected DiffAwareSFTTrainer, got {type(trainer)}"
        )
