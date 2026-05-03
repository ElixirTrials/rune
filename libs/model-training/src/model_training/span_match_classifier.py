"""Span-match failure classifier for `_find_post_in_span` diagnostic.

Classifies every span-match failure into one of seven mutually-exclusive
buckets (TRUNCATION_FRONT, TRUNCATION_TAIL, BPE_DRIFT_START, BPE_DRIFT_END,
BPE_DRIFT_BOTH, WRONG_TURN_LOOKUP, CONTENT_MISMATCH) based on the structural
relationship between ``post_ids`` and the span token window.

Pure functions only — no I/O, no tokeniser, no torch dependency.  Safe to
import on CPU-only machines.

GPU imports are deferred (INFRA-05); this module does not require torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional


class FailureBucket(str, Enum):
    """Seven mutually-exclusive span-match failure categories."""

    TRUNCATION_FRONT = "TRUNCATION_FRONT"
    TRUNCATION_TAIL = "TRUNCATION_TAIL"
    BPE_DRIFT_START = "BPE_DRIFT_START"
    BPE_DRIFT_END = "BPE_DRIFT_END"
    BPE_DRIFT_BOTH = "BPE_DRIFT_BOTH"
    WRONG_TURN_LOOKUP = "WRONG_TURN_LOOKUP"
    CONTENT_MISMATCH = "CONTENT_MISMATCH"


@dataclass
class ClassificationResult:
    """Full diagnostic record for a single span-match failure.

    Attributes:
        bucket: The failure category.
        conv_idx: Conversation index in the dataset.
        span_idx: Index of the span within this conversation.
        turn_idx: Index into post_codes (after keep_end offset).
        span_start: Start position of the span in input_ids.
        span_end: End position of the span in input_ids (exclusive).
        len_post_ids: Length of post_ids (standalone tokenisation).
        len_span_ids: Length of span token window.
        post_ids_head: First up-to-4 token IDs from post_ids.
        span_ids_head: First up-to-4 token IDs of the span window.
        post_ids_tail: Last up-to-4 token IDs from post_ids.
        span_ids_tail: Last up-to-4 token IDs of the span window.
        drift_start_token: Drifted token ID at start boundary (or None).
        drift_end_token: Drifted token ID at end boundary (or None).
        wrong_turn_j: Index j of the other turn that matched (or None).
        lcs_ratio: LCS length / span length (-1.0 if skipped due to length).
        lcs_skipped: True when LCS was not computed (inputs too long).
    """

    bucket: FailureBucket
    conv_idx: int
    span_idx: int
    turn_idx: int
    span_start: int
    span_end: int
    len_post_ids: int
    len_span_ids: int
    post_ids_head: list[int] = field(default_factory=list)
    span_ids_head: list[int] = field(default_factory=list)
    post_ids_tail: list[int] = field(default_factory=list)
    span_ids_tail: list[int] = field(default_factory=list)
    drift_start_token: Optional[int] = None
    drift_end_token: Optional[int] = None
    wrong_turn_j: Optional[int] = None
    lcs_ratio: float = 0.0
    lcs_skipped: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LCS_MAX_LEN = 4096


def _find_subseq(needle: list[int], haystack: list[int]) -> int:
    """Return first offset where ``needle`` is a contiguous run in ``haystack``.

    Returns -1 if not found.

    Args:
        needle: Token-ID sequence to search for.
        haystack: Token-ID sequence to search within.

    Returns:
        First matching offset, or -1.
    """
    n, h = len(needle), len(haystack)
    if n == 0 or n > h:
        return -1
    for off in range(h - n + 1):
        if haystack[off : off + n] == needle:
            return off
    return -1


def _lcs_length(a: list[int], b: list[int]) -> int:
    """Compute the length of the longest common subsequence of ``a`` and ``b``.

    Uses the standard DP algorithm (O(n*m) time and space).  Both inputs are
    capped at ``_LCS_MAX_LEN`` by the caller to prevent pathological runtimes.

    Args:
        a: First token-ID sequence.
        b: Second token-ID sequence.

    Returns:
        LCS length.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[m]


def _make_base(
    bucket: FailureBucket,
    *,
    conv_idx: int,
    span_idx: int,
    turn_idx: int,
    span_start: int,
    span_end: int,
    n_post: int,
    n_span: int,
    post_ids: list[int],
    span_ids: list[int],
) -> ClassificationResult:
    """Construct a :class:`ClassificationResult` with common diagnostic fields."""
    head4_post = post_ids[:4]
    tail4_post = post_ids[-4:] if n_post >= 4 else post_ids[:]
    head4_span = span_ids[:4]
    tail4_span = span_ids[-4:] if n_span >= 4 else span_ids[:]
    return ClassificationResult(
        bucket=bucket,
        conv_idx=conv_idx,
        span_idx=span_idx,
        turn_idx=turn_idx,
        span_start=span_start,
        span_end=span_end,
        len_post_ids=n_post,
        len_span_ids=n_span,
        post_ids_head=list(head4_post),
        span_ids_head=list(head4_span),
        post_ids_tail=list(tail4_post),
        span_ids_tail=list(tail4_span),
    )


def _check_bpe_drift(
    span_ids: list[int],
    post_ids: list[int],
    base: ClassificationResult,
) -> ClassificationResult | None:
    """Check buckets 3–5 (BPE_DRIFT_*).

    Called only when ``len(post_ids) <= len(span_ids)``.

    Args:
        span_ids: Span token window.
        post_ids: Standalone post token IDs.
        base: Pre-constructed result with common fields; mutated via
            :func:`dataclasses.replace` when a bucket fires.

    Returns:
        A :class:`ClassificationResult` for the first matching BPE-drift
        bucket, or ``None`` if none match.
    """
    n_post = len(post_ids)

    # Bucket 3: BPE_DRIFT_START — post[1:] in span AND token before ≠ post[0].
    if n_post >= 2:
        off_start = _find_subseq(post_ids[1:], span_ids)
        if off_start >= 0:
            expected_pre = span_ids[off_start - 1] if off_start > 0 else None
            if expected_pre != post_ids[0]:
                return replace(
                    base,
                    bucket=FailureBucket.BPE_DRIFT_START,
                    drift_start_token=post_ids[0],
                )

    # Bucket 4: BPE_DRIFT_END — post[:-1] in span contiguously.
    if n_post >= 2:
        off_end = _find_subseq(post_ids[:-1], span_ids)
        if off_end >= 0:
            return replace(
                base,
                bucket=FailureBucket.BPE_DRIFT_END,
                drift_end_token=post_ids[-1],
            )

    # Bucket 5: BPE_DRIFT_BOTH — post[1:-1] in span AND neither #3 nor #4 fired.
    if n_post >= 3:
        off_both = _find_subseq(post_ids[1:-1], span_ids)
        if off_both >= 0:
            return replace(
                base,
                bucket=FailureBucket.BPE_DRIFT_BOTH,
                drift_start_token=post_ids[0],
                drift_end_token=post_ids[-1],
            )

    return None


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


def classify_failure(  # noqa: C901
    span_ids: list[int],
    post_ids: list[int],
    span_start: int,
    span_end: int,
    input_ids: list[int],
    all_post_ids_lists: list[list[int]],
    turn_idx: int,
    *,
    conv_idx: int = 0,
    span_idx: int = 0,
) -> ClassificationResult:
    """Classify a single span-match failure into one of seven buckets.

    Decision tree evaluated in order; the first matching predicate wins:

    1. TRUNCATION_FRONT — ``span_start == 0`` AND ``len(post_ids) > len(span_ids)``
       AND there exists k ≥ 1 such that ``post_ids[k:]`` is a prefix of
       ``span_ids`` (suffix-of-post matches prefix-of-span at offset 0).
    2. TRUNCATION_TAIL — ``span_end == len(input_ids)`` AND
       ``len(post_ids) > len(span_ids)`` AND there exists k ≥ 1 such that
       ``post_ids[:-k]`` is found in ``span_ids`` at the tail end.
    3. BPE_DRIFT_START — ``len(post_ids) <= len(span_ids)`` AND
       ``post_ids[1:]`` is a contiguous run inside ``span_ids`` AND the
       token at the position immediately before the match differs from
       ``post_ids[0]`` (or there is no position before it).
    4. BPE_DRIFT_END — ``len(post_ids) <= len(span_ids)`` AND
       ``post_ids[:-1]`` is a contiguous run inside ``span_ids``.
    5. BPE_DRIFT_BOTH — ``len(post_ids) <= len(span_ids)`` AND
       ``post_ids[1:-1]`` is a contiguous run inside ``span_ids`` AND
       neither bucket 3 nor 4 matched.
    6. WRONG_TURN_LOOKUP — none of the above AND another turn's post_ids
       matches the span exactly (full-match via ``_find_subseq`` semantics).
    7. CONTENT_MISMATCH — fallback; reports LCS length / span length ratio.

    Args:
        span_ids: Token IDs of the span window (``input_ids[span_start:span_end]``).
        post_ids: Token IDs from standalone tokenisation of ``post_codes[turn_idx]``.
        span_start: Start position of the span in ``input_ids``.
        span_end: End position of the span in ``input_ids`` (exclusive).
        input_ids: Full token-ID sequence.
        all_post_ids_lists: All per-turn post_ids for this conversation.
            Index ``turn_idx`` is the one being classified; others are
            checked by WRONG_TURN_LOOKUP.
        turn_idx: Index into ``all_post_ids_lists`` for the failing turn.
        conv_idx: Conversation index (metadata for the result).
        span_idx: Span index within this conversation (metadata).

    Returns:
        A :class:`ClassificationResult` with the bucket and diagnostic fields.
    """
    n_post = len(post_ids)
    n_span = len(span_ids)
    total_seq = len(input_ids)

    base = _make_base(
        FailureBucket.CONTENT_MISMATCH,
        conv_idx=conv_idx,
        span_idx=span_idx,
        turn_idx=turn_idx,
        span_start=span_start,
        span_end=span_end,
        n_post=n_post,
        n_span=n_span,
        post_ids=post_ids,
        span_ids=span_ids,
    )

    # Bucket 1: TRUNCATION_FRONT
    if span_start == 0 and n_post > n_span:
        for k in range(1, n_post):
            suffix = post_ids[k:]
            if not suffix:
                break
            if len(suffix) <= n_span and span_ids[: len(suffix)] == suffix:
                return replace(base, bucket=FailureBucket.TRUNCATION_FRONT)

    # Bucket 2: TRUNCATION_TAIL
    if span_end == total_seq and n_post > n_span:
        for k in range(1, n_post):
            prefix = post_ids[:-k] if k < n_post else []
            if not prefix:
                break
            plen = len(prefix)
            if plen <= n_span and span_ids[n_span - plen :] == prefix:
                return replace(base, bucket=FailureBucket.TRUNCATION_TAIL)

    # Buckets 3–5: BPE_DRIFT_* (only when post is no longer than span)
    if n_post <= n_span:
        bpe = _check_bpe_drift(span_ids, post_ids, base)
        if bpe is not None:
            return bpe

    # Bucket 6: WRONG_TURN_LOOKUP
    for j, other_ids in enumerate(all_post_ids_lists):
        if j == turn_idx or not other_ids:
            continue
        if len(other_ids) > n_span:
            continue
        if _find_subseq(other_ids, span_ids) >= 0:
            return replace(base, bucket=FailureBucket.WRONG_TURN_LOOKUP, wrong_turn_j=j)

    # Bucket 7: CONTENT_MISMATCH (fallback)
    if n_post > _LCS_MAX_LEN or n_span > _LCS_MAX_LEN:
        return replace(base, lcs_ratio=-1.0, lcs_skipped=True)
    lcs_len = _lcs_length(post_ids, span_ids)
    lcs_ratio = lcs_len / n_span if n_span > 0 else 0.0
    return replace(base, lcs_ratio=lcs_ratio)
