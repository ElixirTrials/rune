"""Shared scoring core for the recoverability scorecard (Issue #52, spec section 4).

Pure-torch next-token gold-logprob math, lifted VERBATIM from the inner loop of
``tools/diag_recoverability.py`` (``_span_logprob`` / ``_diff_logprob``). It is imported
by BOTH the Qwen probe and the Gemma Doc2LoRA positive control so the two share the
identical numerics -- the entire point of the control: the math is bit-for-bit the same.

Hard constraints (so it loads from the Sakana third_party venv via ``sys.path``):
  - torch + stdlib ONLY; no rune.*, no transformers, no Gemma/Qwen package imports.
  - no relative/package imports.

LOAD-BEARING numeric details (do NOT "clean up"):
  - ``log_softmax(logits.float(), ...)`` -- the float32 cast is part of the shared math.
  - per-token accumulation as ``sum(float(lp[t-1, ids[t]]) for t in ...)`` -- each
    element is cast to a Python double and summed in float64. Vectorizing into
    ``gather(...).sum()`` accumulates in float32 and diverges in the last bits; that
    silently defeats the "identical math" guarantee and a toy test would not catch it.
  - the next-token convention: the logprob of the gold token at position ``t`` is read
    from row ``t-1`` (the distribution that PREDICTS it).

The model forward runs in the CALLER; this core is logits-in. Empty-target / empty-mask
guards (the ``None`` returns in the originals) also stay in the caller -- these
functions assume non-empty inputs and return a ``float``.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def mean_gold_logprob(
    logits: torch.Tensor,
    ids: Sequence[int] | torch.Tensor,
    target_start: int,
    target_len: int,
) -> float:
    """Mean gold-token logprob over the target span that FOLLOWS the prompt.

    Mirrors ``_span_logprob``: ``ids`` is the full 1D sequence ``prompt_ids+target_ids``
    and ``logits`` is the per-sequence ``[L, V]`` tensor (caller passes
    ``model(...).logits[0]``). The target occupies
    ``ids[target_start : target_start + target_len]``; the gold logprob of each target
    token at position ``t`` is read from row ``t-1`` (next-token convention).

    Caller must ensure ``target_len >= 1`` and ``target_start >= 1``.
    """
    lp = torch.log_softmax(logits.float(), dim=-1)
    tot = sum(
        float(lp[t - 1, ids[t]])
        for t in range(target_start, target_start + target_len)
    )
    return tot / target_len


def masked_gold_logprob(
    logits: torch.Tensor,
    ids: Sequence[int] | torch.Tensor,
    mask: Sequence[bool] | torch.Tensor,
) -> float:
    """Mean gold-token logprob over an arbitrary (edit-local) set of positions.

    Mirrors ``_diff_logprob``: ``ids`` is the 1D token sequence (the teacher-forced
    answer, no separate prompt prefix) and ``mask[t]`` selects which positions count.
    Only positions ``t`` in ``range(1, len(ids))`` are eligible -- position 0 is always
    excluded so the ``t-1`` next-token read never wraps to ``lp[-1]`` (the last row).
    The gold logprob of each selected token is read from row ``t-1``.

    Caller must ensure at least one position in ``range(1, len(ids))`` is masked True.
    """
    lp = torch.log_softmax(logits.float(), dim=-1)
    idx = [t for t in range(1, len(ids)) if mask[t]]
    tot = sum(float(lp[t - 1, ids[t]]) for t in idx)
    return tot / len(idx)
