"""Toy-tensor tests for the shared scoring core (Issue #52, spec §4/§8).

No model: tiny hand-built logits/ids, assert the returned logprob equals a hand-computed
value. The crucial bug class is the t-1 next-token off-by-one — the logits are built so that
row t-1 STRONGLY favors the gold token while row t does not, so reading the wrong row yields
a clearly wrong (much lower) number, not a near-miss within tolerance.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from scoring_core import masked_gold_logprob, mean_gold_logprob  # noqa: E402


def _logsoftmax_at(row: list[float], col: int) -> float:
    """Reference log-softmax of a single logits row at one column, in float64."""
    m = max(row)
    denom = sum(math.exp(x - m) for x in row)
    return (row[col] - m) - math.log(denom)


def _build_peaky_logits(vocab: int, peaks: dict[int, int]) -> torch.Tensor:
    """[L, V] logits where row r assigns a big logit to column peaks[r], else ~0.

    Rows not in `peaks` are flat (uniform). A "peak" row makes one token overwhelmingly
    likely, so reading the right vs wrong row gives very different logprobs.
    """
    L = max(peaks) + 2 if peaks else 1
    logits = torch.zeros(L, vocab)
    for r, c in peaks.items():
        logits[r, c] = 10.0
    return logits


def test_mean_gold_logprob_single_target_token():
    # ids = [prompt0, prompt1, gold].  target_start=2, target_len=1.
    # Gold token id = 3 must be read from row t-1 = row 1.
    vocab = 5
    gold = 3
    # row 1 peaks at the gold token (predicts it); row 2 peaks elsewhere (a trap).
    logits = _build_peaky_logits(vocab, {1: gold, 2: 0})
    ids = [4, 4, gold]
    got = mean_gold_logprob(logits, ids, target_start=2, target_len=1)
    expected = _logsoftmax_at(logits[1].tolist(), gold)
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-6)
    # Off-by-one guard: if it had read row 2 instead, the value would be much lower.
    wrong = _logsoftmax_at(logits[2].tolist(), gold)
    assert got - wrong > 5.0


def test_mean_gold_logprob_first_and_last_target_token():
    # ids = [p, g0, g1, g2].  target_start=1, target_len=3 -> targets at t=1,2,3.
    # gold logprob of t read from row t-1: rows 0,1,2.
    vocab = 6
    g0, g1, g2 = 2, 4, 5
    logits = _build_peaky_logits(vocab, {0: g0, 1: g1, 2: g2})
    ids = [1, g0, g1, g2]
    got = mean_gold_logprob(logits, ids, target_start=1, target_len=3)
    # FIRST target token t=1 reads row 0; LAST target token t=3 reads row 2.
    e0 = _logsoftmax_at(logits[0].tolist(), g0)  # t=1, row 0
    e1 = _logsoftmax_at(logits[1].tolist(), g1)  # t=2, row 1
    e2 = _logsoftmax_at(logits[2].tolist(), g2)  # t=3, row 2
    expected = (e0 + e1 + e2) / 3
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-6)


def test_mean_gold_logprob_division_by_target_len():
    # Constant logits -> every gold logprob identical; mean == that value (proves /target_len).
    vocab = 4
    logits = torch.zeros(8, vocab)  # uniform every row -> logprob = -log(vocab)
    ids = [0, 1, 2, 3, 0, 1, 2, 3]
    per_tok = -math.log(vocab)
    for tlen in (1, 3, 5):
        got = mean_gold_logprob(logits, ids, target_start=2, target_len=tlen)
        assert math.isclose(got, per_tok, rel_tol=0, abs_tol=1e-6)


def test_masked_gold_logprob_noncontiguous_mask():
    # ans ids of length 5; mask selects non-contiguous positions t=1 and t=3.
    vocab = 7
    ids = [1, 2, 3, 4, 5]
    # gold token at t=1 is ids[1]=2 read from row 0; at t=3 is ids[3]=4 read from row 2.
    logits = _build_peaky_logits(vocab, {0: 2, 2: 4})
    mask = [False, True, False, True, False]
    got = masked_gold_logprob(logits, ids, mask)
    e_t1 = _logsoftmax_at(logits[0].tolist(), 2)  # t=1 -> row 0
    e_t3 = _logsoftmax_at(logits[2].tolist(), 4)  # t=3 -> row 2
    expected = (e_t1 + e_t3) / 2
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-6)


def test_masked_gold_logprob_ignores_position_zero():
    # mask[0]=True must be ignored (the t>=1 guard prevents lp[-1] wraparound).
    vocab = 5
    ids = [3, 1, 2]
    # Only row 0 -> col ids[1]=1 contributes (t=1). If t=0 were used it would read lp[-1].
    logits = _build_peaky_logits(vocab, {0: 1})
    mask_with_zero = [True, True, False]
    mask_without_zero = [False, True, False]
    got_with = masked_gold_logprob(logits, ids, mask_with_zero)
    got_without = masked_gold_logprob(logits, ids, mask_without_zero)
    assert math.isclose(got_with, got_without, rel_tol=0, abs_tol=1e-9)
    expected = _logsoftmax_at(logits[0].tolist(), 1)
    assert math.isclose(got_with, expected, rel_tol=0, abs_tol=1e-6)


def test_accepts_tensor_ids():
    # ids may be a 1D LongTensor (caller in diag_recoverability passes a tensor row).
    vocab = 5
    gold = 2
    logits = _build_peaky_logits(vocab, {1: gold})
    ids = torch.tensor([4, 4, gold])
    got = mean_gold_logprob(logits, ids, target_start=2, target_len=1)
    expected = _logsoftmax_at(logits[1].tolist(), gold)
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-6)
