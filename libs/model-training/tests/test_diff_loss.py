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

        with patch(
            "model_training.diff_loss.compute_hunk_loss_weights",
            wraps=__import__(
                "model_training.diff_loss", fromlist=["compute_hunk_loss_weights"]
            ).compute_hunk_loss_weights,
        ) as mock_hunk, patch(
            "model_training.diff_loss.compute_diff_loss_weights",
            wraps=__import__(
                "model_training.diff_loss", fromlist=["compute_diff_loss_weights"]
            ).compute_diff_loss_weights,
        ) as mock_legacy:
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
