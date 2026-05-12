"""Unit tests for model_training.span_match_classifier.

Each test constructs synthetic ``(span_ids, post_ids, ...)`` tuples that land
exactly in one of the seven failure buckets, then asserts the classifier
returns the expected bucket.

Extra tests verify:
- Predicate ordering (TRUNCATION_FRONT wins over BPE_DRIFT_START when both
  letter-of-the-law conditions hold simultaneously).
- CONTENT_MISMATCH is the true fallback (returned only when nothing else fires).
- LCS-skipped path triggers when inputs exceed _LCS_MAX_LEN.
"""

from __future__ import annotations

from model_training.span_match_classifier import (
    _LCS_MAX_LEN,
    ClassificationResult,
    FailureBucket,
    classify_failure,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify(
    *,
    span_ids: list[int],
    post_ids: list[int],
    span_start: int,
    input_ids: list[int] | None = None,
    all_post_ids_lists: list[list[int]] | None = None,
    turn_idx: int = 0,
    conv_idx: int = 0,
    span_idx: int = 0,
) -> ClassificationResult:
    """Helper to call classify_failure with convenient defaults."""
    span_end = span_start + len(span_ids)
    if input_ids is None:
        # Default: input_ids is exactly the span (single-span sequence).
        input_ids = span_ids[:]
    if all_post_ids_lists is None:
        all_post_ids_lists = [post_ids]
    return classify_failure(
        span_ids=span_ids,
        post_ids=post_ids,
        span_start=span_start,
        span_end=span_end,
        input_ids=input_ids,
        all_post_ids_lists=all_post_ids_lists,
        turn_idx=turn_idx,
        conv_idx=conv_idx,
        span_idx=span_idx,
    )


# ---------------------------------------------------------------------------
# Bucket 1: TRUNCATION_FRONT
# ---------------------------------------------------------------------------


class TestTruncationFront:
    def test_basic(self) -> None:
        """Span starts at 0, post is longer, suffix of post matches span prefix."""
        # post = [10, 20, 30, 40, 50]  (5 tokens)
        # span = [30, 40, 50]          (3 tokens — first 2 tokens of post truncated)
        # post_ids[2:] == span_ids[:3]  → k=2
        span_ids = [30, 40, 50]
        post_ids = [10, 20, 30, 40, 50]
        # span_start=0, span_end=3, input_ids same as span (len=3)
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.TRUNCATION_FRONT

    def test_one_token_truncated(self) -> None:
        """Single leading token truncated."""
        span_ids = [2, 3, 4, 5]
        post_ids = [1, 2, 3, 4, 5]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.TRUNCATION_FRONT

    def test_not_triggered_when_span_start_nonzero(self) -> None:
        """TRUNCATION_FRONT requires span_start == 0."""
        # If span doesn't start at 0, it should NOT be TRUNCATION_FRONT.
        span_ids = [30, 40, 50]
        post_ids = [10, 20, 30, 40, 50]
        # Build input_ids with a leading token so span_start=1
        input_ids = [99] + span_ids
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=1,
            span_end=4,
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
        )
        assert result.bucket != FailureBucket.TRUNCATION_FRONT


# ---------------------------------------------------------------------------
# Bucket 2: TRUNCATION_TAIL
# ---------------------------------------------------------------------------


class TestTruncationTail:
    def test_basic(self) -> None:
        """Span ends at last position, post is longer, prefix of post at span tail."""
        # post = [10, 20, 30, 40, 50]  (5 tokens)
        # span = [10, 20, 30]          (3 tokens — last 2 tokens of post truncated)
        # post_ids[:-2] == [10,20,30] found at span tail
        span_ids = [10, 20, 30]
        post_ids = [10, 20, 30, 40, 50]
        # span starts at position 5 in a 8-token sequence, ends at 7 (last pos)
        input_ids = [1, 2, 3, 4, 5] + span_ids
        span_start = 5
        span_end = 8
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=span_start,
            span_end=span_end,
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
        )
        assert result.bucket == FailureBucket.TRUNCATION_TAIL

    def test_not_triggered_when_span_end_not_last(self) -> None:
        """TRUNCATION_TAIL requires span_end == len(input_ids)."""
        # span doesn't reach the end of input_ids
        span_ids = [10, 20, 30]
        post_ids = [10, 20, 30, 40, 50]
        # Add trailing token so span_end < len(input_ids)
        input_ids = span_ids + [99]
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            span_end=3,
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
        )
        # Should NOT be TRUNCATION_TAIL
        assert result.bucket != FailureBucket.TRUNCATION_TAIL


# ---------------------------------------------------------------------------
# Bucket 3: BPE_DRIFT_START
# ---------------------------------------------------------------------------


class TestBpeDriftStart:
    def test_basic(self) -> None:
        """post[1:] is in span, but post[0] differs from pre-match token."""
        # span_ids = [100, 20, 30, 40]
        # post_ids = [10, 20, 30, 40]
        # post_ids[1:] = [20, 30, 40] found at offset 1 in span.
        # span_ids[0] = 100 != post_ids[0] = 10  → BPE_DRIFT_START
        span_ids = [100, 20, 30, 40]
        post_ids = [10, 20, 30, 40]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.BPE_DRIFT_START
        assert result.drift_start_token == 10

    def test_drift_at_sequence_boundary(self) -> None:
        """post[1:] matches at offset 0 in span (no preceding token)."""
        # span_ids = [20, 30, 40]
        # post_ids = [10, 20, 30, 40] → too long; skip BPE buckets.
        # post_ids = [10, 20, 30], post[1:] = [20, 30] → found at offset 0.
        # off_start=0, expected_pre = None (no element before offset 0)
        # None != 10 → BPE_DRIFT_START
        span_ids = [20, 30, 40]
        post_ids = [10, 20, 30]
        # Ensure it doesn't satisfy TRUNCATION_FRONT (span_start != 0 is NOT needed;
        # len(post_ids) == len(span_ids) so TRUNCATION_FRONT won't fire).
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.BPE_DRIFT_START


# ---------------------------------------------------------------------------
# Bucket 4: BPE_DRIFT_END
# ---------------------------------------------------------------------------


class TestBpeDriftEnd:
    def test_basic(self) -> None:
        """post[:-1] is found in span; last token drifted."""
        # span_ids = [10, 20, 30, 200]
        # post_ids = [10, 20, 30, 40]
        # post_ids[:-1] = [10, 20, 30] found at offset 0 in span.
        # BPE_DRIFT_START: post_ids[1:] = [20, 30, 40] → not in span
        #   (span has [200] at position 3, not 40). So #3 won't fire.
        # BPE_DRIFT_END fires.
        span_ids = [10, 20, 30, 200]
        post_ids = [10, 20, 30, 40]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.BPE_DRIFT_END
        assert result.drift_end_token == 40

    def test_post_shorter_than_span(self) -> None:
        """post[:-1] found at non-zero offset in span; last token drifted."""
        # span = [99, 10, 20, 30, 200]
        # post = [10, 20, 30, 40]  (post[1:] = [20,30,40] → span has 200 at
        # position 3, not 40 → BPE_DRIFT_START won't fire)
        # post[:-1] = [10, 20, 30] found at offset 1 in span → BPE_DRIFT_END
        span_ids = [99, 10, 20, 30, 200]
        post_ids = [10, 20, 30, 40]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.BPE_DRIFT_END


# ---------------------------------------------------------------------------
# Bucket 5: BPE_DRIFT_BOTH
# ---------------------------------------------------------------------------


class TestBpeDriftBoth:
    def test_basic(self) -> None:
        """post[1:-1] found in span; both boundary tokens drifted."""
        # span = [99, 20, 30, 88]
        # post = [10, 20, 30, 40]
        # post[1:]   = [20, 30, 40] → span has 88 at end, not 40 → no #3
        # post[:-1]  = [10, 20, 30] → span doesn't start with 10 at 0-based
        #   offset; span[0]=99 ≠ 10; try offset 1: span[1:4]=[20,30,88] ≠ [10,20,30]
        #   → no #4
        # post[1:-1] = [20, 30] → found at offset 1 in span → #5
        span_ids = [99, 20, 30, 88]
        post_ids = [10, 20, 30, 40]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.BPE_DRIFT_BOTH
        assert result.drift_start_token == 10
        assert result.drift_end_token == 40


# ---------------------------------------------------------------------------
# Bucket 6: WRONG_TURN_LOOKUP
# ---------------------------------------------------------------------------


class TestWrongTurnLookup:
    def test_other_turn_exact_match(self) -> None:
        """Another turn's post_ids matches the span; correct turn doesn't."""
        # span contains tokens [10, 20, 30]
        # turn_idx=0 has post=[99, 99, 99] → no match
        # turn_idx=1 has post=[10, 20, 30] → exact match in span
        span_ids = [10, 20, 30]
        post_ids_turn0 = [99, 99, 99]
        post_ids_turn1 = [10, 20, 30]
        all_post = [post_ids_turn0, post_ids_turn1]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids_turn0,
            span_start=0,
            input_ids=span_ids,
            all_post_ids_lists=all_post,
            turn_idx=0,
        )
        assert result.bucket == FailureBucket.WRONG_TURN_LOOKUP
        assert result.wrong_turn_j == 1

    def test_no_other_turn_match(self) -> None:
        """When no other turn matches the span, should not return WRONG_TURN_LOOKUP."""
        span_ids = [10, 20, 30]
        post_ids_turn0 = [99, 99, 99]
        post_ids_turn1 = [55, 66, 77]  # doesn't match span
        all_post = [post_ids_turn0, post_ids_turn1]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids_turn0,
            span_start=0,
            input_ids=span_ids,
            all_post_ids_lists=all_post,
            turn_idx=0,
        )
        assert result.bucket != FailureBucket.WRONG_TURN_LOOKUP


# ---------------------------------------------------------------------------
# Bucket 7: CONTENT_MISMATCH
# ---------------------------------------------------------------------------


class TestContentMismatch:
    def test_basic(self) -> None:
        """Completely different tokens — falls through to CONTENT_MISMATCH."""
        # post = [99, 88, 77]  entirely absent from span; no drift patterns,
        # no other-turn match, no truncation signals.
        span_ids = [10, 20, 30, 40]
        post_ids = [99, 88, 77]
        # span_start=2 (not 0), span_end=6, len(input_ids)=10 (not 6)
        input_ids = [1, 2] + span_ids + [5, 6, 7, 8]
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=2,
            span_end=6,
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
        )
        assert result.bucket == FailureBucket.CONTENT_MISMATCH

    def test_lcs_ratio_computed(self) -> None:
        """LCS ratio is set for CONTENT_MISMATCH."""
        span_ids = [10, 20, 30, 40]
        post_ids = [10, 99, 30, 88]  # 2 common tokens
        input_ids = [1, 2] + span_ids + [5, 6]
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=2,
            span_end=6,
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
        )
        assert result.bucket == FailureBucket.CONTENT_MISMATCH
        assert result.lcs_ratio > 0.0
        assert not result.lcs_skipped

    def test_lcs_skipped_when_too_long(self) -> None:
        """LCS is skipped and lcs_ratio=-1 when inputs exceed _LCS_MAX_LEN."""
        # Use a very large span with token IDs in range [0, _LCS_MAX_LEN),
        # and post_ids using tokens way outside that range so no subsequence
        # match can accidentally trigger a BPE_DRIFT bucket.
        base_offset = 1_000_000  # IDs far from span range
        span_ids = list(range(_LCS_MAX_LEN + 1))
        post_ids = [base_offset + i for i in range(500)]  # entirely disjoint IDs
        # span_start=1, span_end must equal len(input_ids) to avoid TRUNC_TAIL
        # and span_start!=0 to avoid TRUNC_FRONT.
        # Use span_start=1, and input_ids=[sentinel] + span_ids (ends there).
        # span_end = 1 + len(span_ids) == len(input_ids), so TRUNCATION_TAIL
        # would require post longer than span AND prefix-at-tail match — our
        # post has completely different IDs so that check won't fire.
        input_ids = [base_offset - 1] + span_ids
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=1,
            span_end=1 + len(span_ids),
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
        )
        assert result.bucket == FailureBucket.CONTENT_MISMATCH
        assert result.lcs_ratio == -1.0
        assert result.lcs_skipped


# ---------------------------------------------------------------------------
# Predicate ordering tests
# ---------------------------------------------------------------------------


class TestPredicateOrdering:
    def test_truncation_front_beats_bpe_drift_start(self) -> None:
        """When both TRUNCATION_FRONT and BPE_DRIFT_START letter-of-the-law hold,
        TRUNCATION_FRONT must win (predicate 1 checked before 3)."""
        # span_start=0, len(post_ids)=5 > len(span_ids)=4 → TRUNCATION_FRONT
        # could also satisfy BPE_DRIFT_START (post[1:] inside span) in theory.
        # Build a case where both fire on letter-of-the-law but order matters.
        # post = [10, 20, 30, 40, 50]; span = [20, 30, 40, 50]
        # TRUNCATION_FRONT: k=1, post[1:] = [20,30,40,50] = span[:4] ✓
        # BPE_DRIFT_START: post[1:] = [20,30,40,50] found at offset 0 in span;
        #   off_start=0, expected_pre=None ≠ post[0]=10 → would also fire #3.
        span_ids = [20, 30, 40, 50]
        post_ids = [10, 20, 30, 40, 50]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.TRUNCATION_FRONT

    def test_bpe_drift_start_checked_before_bpe_drift_end(self) -> None:
        """BPE_DRIFT_START (bucket 3) is checked before BPE_DRIFT_END is even evaluated.

        In this input BPE_DRIFT_END would not fire anyway (post[:-1]=[10,20,30]
        is not present in span=[100,20,30,40]), so this test does not exercise
        simultaneous satisfaction. It verifies only that BPE_DRIFT_START fires
        when its predicate is met, regardless of bucket 4.
        """
        # span = [100, 20, 30, 40];  post = [10, 20, 30, 40]
        # post[1:]  = [20, 30, 40] found at offset 1 in span;
        #   expected_pre = span[0] = 100 ≠ 10 → #3 fires.
        # post[:-1] = [10, 20, 30] → not found in span → #4 would not fire.
        span_ids = [100, 20, 30, 40]
        post_ids = [10, 20, 30, 40]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.BPE_DRIFT_START

    def test_bpe_drift_end_beats_bpe_drift_both(self) -> None:
        """BPE_DRIFT_END (bucket 4) fires before BPE_DRIFT_BOTH (bucket 5)."""
        # span = [10, 20, 30, 200];  post = [10, 20, 30, 40]
        # post[1:]  = [20, 30, 40] → span[1:4]=[20,30,200] ≠ → #3 doesn't fire
        # post[:-1] = [10, 20, 30] → found at offset 0 in span → #4 fires.
        span_ids = [10, 20, 30, 200]
        post_ids = [10, 20, 30, 40]
        result = _classify(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=0,
            input_ids=span_ids,
        )
        assert result.bucket == FailureBucket.BPE_DRIFT_END

    def test_wrong_turn_lookup_beats_content_mismatch(self) -> None:
        """WRONG_TURN_LOOKUP fires before CONTENT_MISMATCH."""
        span_ids = [10, 20, 30]
        wrong_post = [99, 99, 99]  # failing turn
        other_post = [10, 20, 30]  # another turn matches the span
        all_post = [wrong_post, other_post]
        # span_start nonzero so no TRUNCATION_FRONT; post shorter than span?
        # len(wrong_post)=3 == len(span_ids)=3 → BPE checks eligible but [99,99,99]
        # vs [10,20,30] → no inner-sequence match.
        input_ids = [1] + span_ids + [2]
        result = classify_failure(
            span_ids=span_ids,
            post_ids=wrong_post,
            span_start=1,
            span_end=4,
            input_ids=input_ids,
            all_post_ids_lists=all_post,
            turn_idx=0,
        )
        assert result.bucket == FailureBucket.WRONG_TURN_LOOKUP
        assert result.wrong_turn_j == 1


# ---------------------------------------------------------------------------
# Metadata fields on result
# ---------------------------------------------------------------------------


class TestMetadataFields:
    def test_conv_span_idx_passed_through(self) -> None:
        """conv_idx and span_idx should appear in the result."""
        span_ids = [10, 20, 30, 40]
        post_ids = [99, 88, 77]
        input_ids = [1, 2] + span_ids + [5, 6]
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=2,
            span_end=6,
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
            conv_idx=42,
            span_idx=7,
        )
        assert result.conv_idx == 42
        assert result.span_idx == 7

    def test_head_tail_populated(self) -> None:
        """post_ids_head / span_ids_head are populated even in CONTENT_MISMATCH."""
        span_ids = [10, 20, 30, 40]
        post_ids = [99, 88, 77]
        input_ids = [1, 2] + span_ids + [5, 6]
        result = classify_failure(
            span_ids=span_ids,
            post_ids=post_ids,
            span_start=2,
            span_end=6,
            input_ids=input_ids,
            all_post_ids_lists=[post_ids],
            turn_idx=0,
        )
        assert result.post_ids_head == [99, 88, 77]
        assert result.span_ids_head == [10, 20, 30, 40]
