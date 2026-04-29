"""Line-level diff-aware loss weighting for SFT training.

GPU-heavy imports (torch, transformers) are deferred inside function bodies to
ensure CPU-only importability (INFRA-05).

trl.SFTTrainer is imported at module top-level with a try/except guard so that
``DiffAwareSFTTrainer`` is importable on CPU-only machines (it falls back to
subclassing ``object`` when trl is absent).
"""

from __future__ import annotations

import difflib
import logging
from typing import Any

try:
    from trl import SFTTrainer  # type: ignore[attr-defined]
except ModuleNotFoundError:
    # ModuleNotFoundError only — broken trl installs should surface loudly.
    SFTTrainer = object  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

IGNORE_INDEX: int = -100

# Module-level flag to emit the hunk-fallback warning only once.
_HUNK_FALLBACK_WARNED: bool = False


# ---------------------------------------------------------------------------
# Hunk-path helpers (new, line-level diff engine)
# ---------------------------------------------------------------------------


def _compute_hunk_ranges(before: str, after: str) -> list[tuple[int, int]]:
    """Return character ranges in ``after`` that correspond to + / replace hunks.

    Uses ``difflib.SequenceMatcher`` on lines (``str.splitlines(keepends=True)``).
    Opcodes with tag in ``{"insert", "replace"}`` contribute a half-open char
    range ``(char_start, char_end)`` in the *after* string.  ``"equal"`` and
    ``"delete"`` opcodes contribute nothing.

    Args:
        before: Original source text.
        after: Modified source text.

    Returns:
        List of ``(start, end)`` char offsets (half-open), sorted ascending.
        Empty list when ``before == after``.
    """
    if before == after:
        return []

    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    # Build prefix-sum over after_lines so we can map line index → char offset.
    after_prefix: list[int] = [0]
    for line in after_lines:
        after_prefix.append(after_prefix[-1] + len(line))

    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    ranges: list[tuple[int, int]] = []

    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            char_start = after_prefix[j1]
            char_end = after_prefix[j2]
            if char_end > char_start:
                ranges.append((char_start, char_end))

    return ranges


def compute_hunk_loss_weights(
    input_ids: list[int],
    labels: list[int],
    offset_mapping: list[tuple[int, int]],
    hunk_ranges_chars: list[tuple[int, int]],
    *,
    changed_weight: float = 1.0,
    unchanged_weight: float = 0.3,
) -> list[float]:
    """Map hunk char ranges to tokens via offset_mapping; produce per-token weights.

    Weight assignment rules:

    - Masked positions (``labels[i] == IGNORE_INDEX``) → ``0.0``.
    - Special tokens where ``offset_mapping[i] == (0, 0)`` → ``0.0`` (BOS,
      padding, etc.).
    - Assistant-turn tokens whose offset intersects any hunk range
      → ``changed_weight``.
    - Assistant-turn tokens outside all hunk ranges → ``unchanged_weight``.

    Token *i* intersects hunk ``[s, e)`` iff ``offset_mapping[i] == (a, b)``
    satisfies ``a < e and b > s``.

    Identity invariant: when ``changed_weight == unchanged_weight``, every
    non-masked, non-special assistant token gets that single weight.

    Args:
        input_ids: Token IDs for the full sequence.
        labels: Label IDs; ``IGNORE_INDEX`` marks non-assistant positions.
        offset_mapping: ``(start, end)`` char offsets for each token in the
            source text.  ``(0, 0)`` indicates a special/padding token.
        hunk_ranges_chars: Half-open ``(start, end)`` char ranges from
            ``_compute_hunk_ranges``.
        changed_weight: Weight for tokens inside a hunk range.
        unchanged_weight: Weight for assistant tokens outside any hunk range.

    Returns:
        Per-token float weights of the same length as ``input_ids``.
    """
    n = len(input_ids)
    weights: list[float] = []

    for i in range(n):
        label = labels[i] if i < len(labels) else IGNORE_INDEX
        if label == IGNORE_INDEX:
            weights.append(0.0)
            continue

        om = offset_mapping[i] if i < len(offset_mapping) else (0, 0)
        tok_start, tok_end = om

        # Special / padding tokens emitted as (0, 0) by the tokenizer.
        if tok_start == 0 and tok_end == 0:
            weights.append(0.0)
            continue

        # Check intersection with any hunk range.
        in_hunk = any(
            tok_start < h_end and tok_end > h_start
            for h_start, h_end in hunk_ranges_chars
        )
        weights.append(changed_weight if in_hunk else unchanged_weight)

    return weights


# ---------------------------------------------------------------------------
# Legacy bag-of-token-ids path (deprecated, kept as fallback)
# ---------------------------------------------------------------------------


def compute_diff_loss_weights(
    input_ids: list[int],
    labels: list[int],
    changed_ids: set[int],
    *,
    changed_weight: float = 1.0,
    unchanged_weight: float = 0.3,
) -> list[float]:
    """Compute per-token loss weights based on a set of changed token IDs.

    .. deprecated::
        Prefer :func:`compute_hunk_loss_weights` with ``pre_code``/``post_code``
        side-channels.  This set-based fallback cannot distinguish *where* a
        token appears, only *whether* its ID was seen in the diff.

    Args:
        input_ids: Token IDs for the full sequence.
        labels: Label IDs; ``IGNORE_INDEX`` marks non-assistant positions.
        changed_ids: Set of token IDs considered "changed" by the diff.
        changed_weight: Weight assigned to tokens whose ID is in
            ``changed_ids``.
        unchanged_weight: Weight assigned to non-masked tokens whose ID is
            not in ``changed_ids``.

    Returns:
        Per-token float weights of the same length as ``input_ids``.
    """
    weights: list[float] = []
    for token_id, label in zip(input_ids, labels):
        if label == IGNORE_INDEX:
            weights.append(0.0)
        elif token_id in changed_ids:
            weights.append(changed_weight)
        else:
            weights.append(unchanged_weight)
    return weights


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------


def _iter_assistant_spans(
    labels: list[int],
) -> list[tuple[int, int]]:
    """Yield half-open ``(start, end)`` ranges of contiguous label-bearing tokens.

    A span is a maximal run of positions where ``label != IGNORE_INDEX``.
    Used by the diff-aware collator to recover per-assistant-turn boundaries
    from the labels tensor that the inner ``DataCollatorForLanguageModeling``
    produces (which already merged ``assistant_masks`` into the label mask).
    """
    spans: list[tuple[int, int]] = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == IGNORE_INDEX:
            i += 1
            continue
        start = i
        while i < n and labels[i] != IGNORE_INDEX:
            i += 1
        spans.append((start, i))
    return spans


def _find_post_in_span(
    input_ids_seq: list[int],
    span_start: int,
    span_end: int,
    post_input_ids: list[int],
) -> int:
    """Return the local offset where ``post_input_ids`` matches inside the span.

    The span's ``input_ids`` typically wrap the post body in chat-template
    role markers and a section header (``## Revision`` / ``## Implementation``),
    so the post tokens land somewhere *inside* the span.  Returns ``-1`` when
    no contiguous match exists or the post body is empty / longer than the
    span.

    Linear subsequence search; span length is bounded by SFT ``max_length``
    (~2k) and ``len(post_input_ids)`` by the per-turn body, so worst case is
    well under a millisecond per span — and only runs once per training step.
    """
    n_post = len(post_input_ids)
    span_len = span_end - span_start
    if not (0 < n_post <= span_len):
        return -1
    span_ids = input_ids_seq[span_start:span_end]
    for off in range(span_len - n_post + 1):
        if span_ids[off : off + n_post] == post_input_ids:
            return off
    return -1


def _fill_identity(weights: list[float], start: int, end: int) -> None:
    """Set ``weights[start:end] = 1.0`` so the span keeps gradient signal."""
    for j in range(start, end):
        weights[j] = 1.0


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


class DiffWeightedDataCollator:
    """Data collator that applies diff-aware loss weights to SFT batches.

    Wraps an inner collator (e.g. ``trl.DataCollatorForCompletionOnlyLM``).
    On each ``__call__``, the inner collator runs first to produce
    ``batch["input_ids"]`` and ``batch["labels"]``.  Diff weights are then
    computed and stored as ``batch["loss_weights"]``.

    Hunk path (preferred):
        When all features carry per-turn ``pre_codes`` / ``post_codes``
        list side-channels AND a ``tokenizer`` is provided, weights are
        computed by walking each contiguous assistant span (positions
        where ``label != IGNORE_INDEX``) and aligning it against its own
        per-turn ``(pre_k, post_k)`` pair.  The tokenizer re-tokenizes each
        turn's ``post_k`` with ``return_offsets_mapping=True`` to obtain
        char-to-token alignment, and a subsequence search inside the span
        finds where the post-revision body sits.

        Pre-2026 legacy: a single joined ``pre_code`` / ``post_code`` string
        per feature.  Still accepted (wrapped into a length-1 list) so
        old callers keep working, but multi-turn datasets MUST use the
        per-turn lists — joined strings cannot be aligned against a
        chat-templated multi-turn sequence and silently zero out every
        weight on those rows (RCA-5 H2 regression).

    Fallback path:
        When any feature is missing pre/post side-channels or no
        tokenizer was supplied, the collator emits identity weights
        (``1.0`` for labeled tokens, ``0.0`` for masked) and warns once.
        Spans that fail per-turn alignment also collapse to identity so
        gradient signal is preserved even when hunk weighting is lost.

    Args:
        inner_collator: The underlying data collator.
        changed_weight: Weight for changed / inserted tokens.
        unchanged_weight: Weight for equal (context) tokens.
        tokenizer: Optional tokenizer.  Required to use the hunk path.
    """

    def __init__(
        self,
        inner_collator: Any,
        *,
        changed_weight: float = 1.0,
        unchanged_weight: float = 0.3,
        tokenizer: Any | None = None,
    ) -> None:
        """Initialise the collator.

        Args:
            inner_collator: The underlying data collator.
            changed_weight: Weight for changed / inserted tokens.
            unchanged_weight: Weight for unchanged (context) tokens.
            tokenizer: Optional tokenizer; enables the hunk path when set.
        """
        self.inner_collator = inner_collator
        self.changed_weight = changed_weight
        self.unchanged_weight = unchanged_weight
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weights_via_hunk_path(
        self,
        input_ids_seq: list[int],
        labels_seq: list[int],
        pre_codes: list[str],
        post_codes: list[str],
    ) -> list[float]:
        """Compute weights with per-turn hunk alignment.

        Walks ``labels_seq`` to find each contiguous assistant span (a run
        of ``label != IGNORE_INDEX`` positions, one per assistant turn),
        then for each span looks up its corresponding ``(pre_k, post_k)``
        from the per-turn lists by index — `keep_start` truncation can
        leave fewer surviving spans than turns, but the surviving ones
        are always the first K, so positional pairing is correct.

        Per-span heavy lifting lives in :meth:`_apply_span_weights`.  When
        alignment fails (or ``post_k`` is empty), the span collapses to
        identity weights so gradient signal is preserved — this is the
        RCA-5 H2 zero-gradient regression guard.

        Args:
            input_ids_seq: Full sequence token IDs.
            labels_seq: Full sequence labels.
            pre_codes: Per-turn pre-revision code bodies.
            post_codes: Per-turn post-revision code bodies.

        Returns:
            Per-token float weights of length ``len(input_ids_seq)``.
        """
        assert self.tokenizer is not None  # guaranteed by caller; satisfies mypy

        weights: list[float] = [0.0] * len(labels_seq)

        for idx, (span_start, span_end) in enumerate(_iter_assistant_spans(labels_seq)):
            pre = pre_codes[idx] if idx < len(pre_codes) else ""
            post = post_codes[idx] if idx < len(post_codes) else ""
            self._apply_span_weights(
                weights, input_ids_seq, span_start, span_end, pre, post
            )

        return weights

    def _apply_span_weights(
        self,
        weights: list[float],
        input_ids_seq: list[int],
        span_start: int,
        span_end: int,
        pre: str,
        post: str,
    ) -> None:
        """Fill ``weights[span_start:span_end]`` for one assistant turn.

        Falls back to identity (``1.0``) when no ``post`` is available, no
        subsequence match is found, or every matched token would otherwise
        end up at weight ``0`` (e.g. tokenizer emitted only special-token
        offsets).  This is the per-span half of the RCA-5 H2 guard.
        """
        if not post:
            _fill_identity(weights, span_start, span_end)
            return

        assert self.tokenizer is not None
        enc = self.tokenizer(
            post,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        post_offsets: list[tuple[int, int]] = [
            tuple(o)
            for o in enc["offset_mapping"]  # type: ignore[misc]
        ]
        post_input_ids: list[int] = list(enc["input_ids"])

        match_pos = _find_post_in_span(
            input_ids_seq, span_start, span_end, post_input_ids
        )
        if match_pos < 0:
            _fill_identity(weights, span_start, span_end)
            return

        hunk_ranges = _compute_hunk_ranges(pre, post)
        n_post = len(post_input_ids)
        for j in range(span_start, span_end):
            local = j - span_start
            if local < match_pos or local >= match_pos + n_post:
                continue
            ts, te = post_offsets[local - match_pos]
            if ts == 0 and te == 0:
                continue
            in_hunk = any(ts < h_end and te > h_start for h_start, h_end in hunk_ranges)
            weights[j] = self.changed_weight if in_hunk else self.unchanged_weight

        # Per-span safety net: an all-special turn would still leave every
        # weight at zero, so restore identity inside the span.
        if all(weights[j] == 0.0 for j in range(span_start, span_end)):
            _fill_identity(weights, span_start, span_end)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of feature dicts into a training batch with loss weights.

        Side-channel columns ``pre_codes`` / ``post_codes`` (per-turn lists)
        are popped from each feature before passing to the inner collator
        so they don't interfere with tensor stacking.  Legacy singular
        ``pre_code`` / ``post_code`` strings are accepted as length-1 lists
        for backward compatibility.

        Args:
            features: List of example dicts.  Each feature may carry either
                per-turn lists (``pre_codes`` / ``post_codes``) or, for
                backwards compatibility, a single ``pre_code`` / ``post_code``
                string that's wrapped into a length-1 list.

        Returns:
            Batch dict with standard keys plus ``"loss_weights"`` tensor.
        """
        global _HUNK_FALLBACK_WARNED

        # Pop side-channel columns before handing to the inner collator.
        # Per-turn lists are preferred; legacy singular keys are wrapped
        # so single-turn callers still work without code changes.
        pre_codes_batch: list[list[str] | None] = []
        post_codes_batch: list[list[str] | None] = []
        for feat in features:
            pre_list = feat.pop("pre_codes", None)
            post_list = feat.pop("post_codes", None)
            legacy_pre = feat.pop("pre_code", None)
            legacy_post = feat.pop("post_code", None)
            if pre_list is None and legacy_pre is not None:
                pre_list = [legacy_pre]
            if post_list is None and legacy_post is not None:
                post_list = [legacy_post]
            pre_codes_batch.append(pre_list)
            post_codes_batch.append(post_list)

        batch = self.inner_collator(features)

        # Determine which path to use.
        has_side_channels = all(p is not None for p in pre_codes_batch) and all(
            p is not None for p in post_codes_batch
        )
        use_hunk_path = has_side_channels and self.tokenizer is not None

        if not use_hunk_path and not _HUNK_FALLBACK_WARNED:
            logger.warning(
                "DiffWeightedDataCollator: pre_codes/post_codes side-channels "
                "missing or tokenizer not set; falling back to identity loss "
                "weights (1.0 for labeled tokens, 0.0 otherwise). "
                "Pass tokenizer= and include per-turn pre_codes/post_codes lists "
                "in dataset features to use the hunk path."
            )
            _HUNK_FALLBACK_WARNED = True

        # Defer torch import — must stay importable in CPU-only CI.
        import torch

        input_ids_batch: Any = batch["input_ids"]
        labels_batch: Any = batch["labels"]

        all_weights: list[list[float]] = []
        for idx in range(len(features)):
            ids_seq: list[int] = input_ids_batch[idx].tolist()
            lab_seq: list[int] = labels_batch[idx].tolist()
            pre_list = pre_codes_batch[idx] or []
            post_list = post_codes_batch[idx] or []

            if use_hunk_path:
                w = self._weights_via_hunk_path(
                    ids_seq, lab_seq, list(pre_list), list(post_list)
                )
            else:
                # Identity fallback: 1.0 for labeled tokens, 0.0 for IGNORE_INDEX.
                # Avoids silently reducing the training objective to a uniform
                # rescale when the diff cannot be computed (no side-channels
                # or no tokenizer).
                w = [1.0 if lab != IGNORE_INDEX else 0.0 for lab in lab_seq]
            all_weights.append(w)

        batch["loss_weights"] = torch.tensor(all_weights, dtype=torch.float32)
        return batch


# ---------------------------------------------------------------------------
# Pure weighted-loss helper (no Trainer base dependency — unit-testable)
# ---------------------------------------------------------------------------


def _compute_weighted_loss(
    logits: Any,
    labels: Any,
    loss_weights: Any,
) -> Any:
    """Compute weighted per-token cross-entropy loss.

    Applies the causal-LM shift (``logits[:, :-1]`` vs ``labels[:, 1:]``),
    masks IGNORE_INDEX positions, and returns the weighted mean:

        L = sum(per_token_CE * w * mask) / sum(w * mask)

    This reduces to standard mean CE when all ``loss_weights == 1.0``.

    Args:
        logits: Float tensor of shape ``[batch, seq_len, vocab_size]``.
        labels: Long tensor of shape ``[batch, seq_len]`` with
            ``IGNORE_INDEX`` (``-100``) for masked positions.
        loss_weights: Float tensor of shape ``[batch, seq_len]``.

    Returns:
        Scalar loss tensor.
    """
    import torch.nn.functional as F  # noqa: N812

    # Causal-LM shift: predict token t+1 from hidden state at t.
    shift_logits = logits[:, :-1, :].contiguous()  # [B, S-1, V]
    shift_labels = labels[:, 1:].contiguous()  # [B, S-1]
    shift_weights = loss_weights[:, 1:].contiguous()  # [B, S-1]

    # Per-token CE with no internal reduction.
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.shape)  # [B, S-1]

    label_mask = (shift_labels != IGNORE_INDEX).float()  # [B, S-1]
    weighted = per_token_loss * shift_weights * label_mask
    denom = (shift_weights * label_mask).sum()

    # Guard: all-masked batch (no labeled/weighted tokens).  Clamping at
    # 1e-8 prevents NaN; after Task 1's masking fix this can only fire on a
    # genuinely degenerate batch — operators must see it (RCA-5 visibility).
    if denom.item() < 1e-8:
        # Visible at WARNING because after Task 1's masking fix, this can
        # only fire on a genuinely degenerate batch — operators must see it.
        logger.warning(
            "DiffAwareSFTTrainer: all-masked batch (denom=%.3e); "
            "weighted loss clamped to 1e-8. If this fires every step, "
            "training has zero gradient signal (RCA-5 H2 regression).",
            denom.item(),
        )
    weighted_loss = weighted.sum() / denom.clamp(min=1e-8)
    return weighted_loss


# ---------------------------------------------------------------------------
# Per-step diagnostic metrics (no extra forward pass)
# ---------------------------------------------------------------------------


def _compute_step_metrics(
    logits: Any,
    labels: Any,
    loss_weights: Any,
    *,
    changed_weight: float,
    unchanged_weight: float,
) -> dict[str, float]:
    """Per-step training observability metrics from the existing forward pass.

    Computed under ``torch.no_grad`` so autograd doesn't retain the
    intermediate softmax / probability tensors.  Returns scalar Python
    floats so the trainer can accumulate them across micro-batches before
    emission via ``Trainer.log``.

    Metrics (all train-side, prefixed ``train/`` by the trainer):

    - ``effective_token_count``: ``(loss_weights * label_mask).sum()`` —
      the loss denominator.  Drops to 0 when the all-masked-batch guard
      fires; pair with ``all_masked_batch`` to detect the RCA-5 H2 path.
    - ``all_masked_batch``: 1.0 when ``effective_token_count < 1e-8``,
      else 0.0.  Averaged across the logging window this becomes the
      fraction of zero-gradient micro-batches.
    - ``token_accuracy``: top-1 ``argmax(logits) == label`` over labeled
      tokens.
    - ``entropy``: mean predictive entropy over labeled tokens.
    - ``changed_loss`` / ``context_loss``: per-token CE split by the
      diff weight class (changed = hunk tokens, context = non-hunk
      assistant tokens).  When ``changed_weight == unchanged_weight``
      (identity fallback), context is empty and everything counts as
      changed.
    - ``changed_token_acc`` / ``context_token_acc``: top-1 accuracy split
      the same way.
    - ``changed_entropy`` / ``context_entropy``: mean entropy split.
    - ``changed_token_frac``: fraction of labeled tokens classed as
      "changed" — drift here means dataset shuffling is non-uniform
      across hunk-density.

    Args:
        logits: ``[batch, seq_len, vocab]`` float tensor (pre-shift).
        labels: ``[batch, seq_len]`` long tensor with ``IGNORE_INDEX`` for
            non-assistant positions.
        loss_weights: ``[batch, seq_len]`` float tensor aligned with
            ``labels``.
        changed_weight: Per-token weight assigned to changed (hunk)
            tokens by the diff collator.
        unchanged_weight: Per-token weight for non-hunk assistant tokens.

    Returns:
        Dict of scalar Python floats; keys are unprefixed metric names.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    with torch.no_grad():
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = loss_weights[:, 1:].contiguous().float()

        label_mask = shift_labels != IGNORE_INDEX
        n_labeled = int(label_mask.sum().item())

        # Effective token count + all-masked-batch flag (the RCA-5 H2 watchdog).
        eff_count = float((shift_weights * label_mask.float()).sum().item())
        all_masked = 1.0 if eff_count < 1e-8 else 0.0

        if n_labeled == 0:
            # Degenerate batch — emit zeros for every other metric.
            zero_keys = (
                "token_accuracy",
                "entropy",
                "changed_loss",
                "context_loss",
                "changed_token_acc",
                "context_token_acc",
                "changed_entropy",
                "context_entropy",
                "changed_token_frac",
            )
            out: dict[str, float] = dict.fromkeys(zero_keys, 0.0)
            out["effective_token_count"] = eff_count
            out["all_masked_batch"] = all_masked
            return out

        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(shift_labels.shape)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

        pred = shift_logits.argmax(dim=-1)
        correct = (pred == shift_labels) & label_mask
        correct_f = correct.float()

        # Split changed vs context by the midpoint between the two diff
        # weights.  When changed == unchanged (identity fallback), the
        # midpoint test collapses both to "changed" — context is empty,
        # which is the correct semantic since there's no diff signal.
        if changed_weight == unchanged_weight:
            changed_mask = label_mask
            context_mask = torch.zeros_like(label_mask)
        else:
            midpoint = (changed_weight + unchanged_weight) / 2.0
            changed_mask = label_mask & (shift_weights >= midpoint)
            context_mask = label_mask & (shift_weights > 0) & (shift_weights < midpoint)

        n_changed = int(changed_mask.sum().item())
        n_context = int(context_mask.sum().item())

        def _masked_mean(t: Any, mask: Any, n: int) -> float:
            if n == 0:
                return 0.0
            return float((t * mask.float()).sum().item()) / n

        return {
            "effective_token_count": eff_count,
            "all_masked_batch": all_masked,
            "token_accuracy": float(correct_f.sum().item()) / n_labeled,
            "entropy": _masked_mean(entropy, label_mask, n_labeled),
            "changed_loss": _masked_mean(per_token_ce, changed_mask, n_changed),
            "context_loss": _masked_mean(per_token_ce, context_mask, n_context),
            "changed_token_acc": _masked_mean(correct_f, changed_mask, n_changed),
            "context_token_acc": _masked_mean(correct_f, context_mask, n_context),
            "changed_entropy": _masked_mean(entropy, changed_mask, n_changed),
            "context_entropy": _masked_mean(entropy, context_mask, n_context),
            "changed_token_frac": float(n_changed) / n_labeled,
        }


# ---------------------------------------------------------------------------
# DiffAwareSFTTrainer
# ---------------------------------------------------------------------------


class DiffAwareSFTTrainer(SFTTrainer):  # type: ignore[misc,valid-type]
    """SFTTrainer that applies per-token loss weights from the batch.

    Expects the data collator (e.g. ``DiffWeightedDataCollator``) to populate
    ``batch["loss_weights"]`` as a ``Tensor[batch, seq_len]`` float32 aligned
    with ``labels``.  The weighted loss is:

        L = sum(per_token_CE * loss_weights * label_mask) /
            sum(loss_weights * label_mask)

    where ``label_mask = (labels != IGNORE_INDEX)``.  This reduces to the
    standard CE loss when all ``loss_weights == 1.0``, so the
    identity-under-uniform-weights invariant holds.

    When ``loss_weights`` is absent from the batch, falls back to standard
    SFTTrainer loss (no ``KeyError``).

    Per-step diagnostic metrics — `changed_loss` / `context_loss`,
    token_accuracy, entropy, effective_token_count, all_masked_batch_frac
    — are accumulated across micro-batches in ``compute_loss`` and flushed
    via ``log`` at the parent's ``logging_steps`` cadence.  See
    :func:`_compute_step_metrics` for the per-batch derivation.  The
    ``_diff_changed_weight`` / ``_diff_unchanged_weight`` instance
    attributes are populated by :func:`build_diff_aware_sft_trainer`; they
    default to the same values used by the collator so the changed/context
    split remains consistent.
    """

    # Defaults match build_diff_aware_sft_trainer's kwargs so a manually
    # constructed instance still produces sensible metric splits.
    _diff_changed_weight: float = 1.0
    _diff_unchanged_weight: float = 0.3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise per-step metric accumulators alongside the parent state."""
        super().__init__(*args, **kwargs)
        self._diff_metric_sums: dict[str, float] = {}
        self._diff_metric_count: int = 0

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Compute weighted cross-entropy loss and accumulate step metrics.

        Pops ``loss_weights`` from ``inputs`` before the forward pass so the
        model never receives an unexpected keyword argument.  When
        ``loss_weights`` is absent, delegates to the parent implementation.

        Args:
            model: The language model.
            inputs: Batch dict (possibly containing ``"loss_weights"``).
            return_outputs: If ``True``, return ``(loss, outputs)`` tuple.
            num_items_in_batch: Passed through to parent when falling back.

        Returns:
            Scalar loss, or ``(loss, outputs)`` when ``return_outputs=True``.
        """
        loss_weights = inputs.pop("loss_weights", None)

        if loss_weights is None:
            # No weights provided — fall back to the model's own CE loss.
            # HuggingFace causal-LM heads honor -100 label masking internally,
            # so outputs.loss is standard CE on the labeled tokens. Avoids
            # super().compute_loss() which depends on full Trainer init state
            # (self.model, self.processing_class, …) and can't be exercised
            # from a minimal subclass used in unit tests.
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Move loss_weights to the same device as logits.
        loss_weights = loss_weights.to(logits.device)

        loss = _compute_weighted_loss(logits, labels, loss_weights)

        # Accumulate per-batch metrics for emission via log().  Wrapped in
        # try/except because metrics are observability-only — a bug here
        # must never kill a training run.
        try:
            metrics = _compute_step_metrics(
                logits,
                labels,
                loss_weights,
                changed_weight=self._diff_changed_weight,
                unchanged_weight=self._diff_unchanged_weight,
            )
            for k, v in metrics.items():
                self._diff_metric_sums[k] = self._diff_metric_sums.get(k, 0.0) + v
            self._diff_metric_count += 1
        except Exception:  # noqa: BLE001
            logger.exception(
                "DiffAwareSFTTrainer: step-metrics accumulation failed; "
                "training continues without per-step diagnostics for this batch."
            )

        return (loss, outputs) if return_outputs else loss

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        """Flush accumulated per-step metrics into ``logs`` before parent emits.

        Detects whether ``logs`` is an eval-context dict by the presence of
        ``eval_loss`` (HuggingFace Trainer's canonical eval-loss key) and
        prefixes the accumulated metrics as ``eval/<key>`` instead of
        ``train/<key>``. This lets downstream callbacks like
        ``OptunaScreeningCallback`` read smoothed per-step metrics from the
        eval-side dict without requiring a separate forward pass.
        """
        if self._diff_metric_count > 0:
            count = self._diff_metric_count
            prefix = "eval" if "eval_loss" in logs else "train"
            for key, total in self._diff_metric_sums.items():
                logs[f"{prefix}/{key}"] = total / count
            # Promote `all_masked_batch` mean (0/1 per call) to a clearer
            # name so dashboards show it as a fraction.
            mean_key = f"{prefix}/all_masked_batch"
            if mean_key in logs:
                logs[f"{prefix}/all_masked_batch_frac"] = logs.pop(mean_key)
            self._diff_metric_sums = {}
            self._diff_metric_count = 0
        return super().log(logs, start_time)


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------


def build_diff_aware_sft_trainer(
    model: Any,
    args: Any,
    train_dataset: Any,
    *,
    peft_config: Any | None = None,
    processing_class: Any | None = None,
    changed_weight: float = 1.0,
    unchanged_weight: float = 0.3,
    tokenizer: Any | None = None,
) -> Any:
    """Build an SFTTrainer that uses diff-aware loss weighting.

    Constructs a :class:`DiffWeightedDataCollator` wrapping trl's
    ``DataCollatorForLanguageModeling`` (completion-only masking via
    ``completion_mask``/``assistant_masks`` set upstream by SFTTrainer's
    chat-template preprocessing), then instantiates
    :class:`DiffAwareSFTTrainer` with it.

    The ``tokenizer`` kwarg enables the hunk (line-level diff) path in the
    collator.  When ``None`` (default), the collator falls back to the legacy
    set-based path.  Task 5 threads the tokenizer through
    ``trainer.py::_construct_sft_trainer``; until then, callers may pass it
    explicitly here.

    All GPU imports are deferred to this function body (INFRA-05).

    Args:
        model: The language model (or PEFT-wrapped model).
        args: ``trl.SFTConfig`` training arguments.
        train_dataset: HuggingFace dataset with SFT examples.
        peft_config: Optional PEFT config for fresh LoRA init.
        processing_class: Tokenizer / processor for the SFTTrainer.
        changed_weight: Weight for changed / inserted tokens.
        unchanged_weight: Weight for unchanged (context) tokens.
        tokenizer: Optional tokenizer passed to :class:`DiffWeightedDataCollator`
            to enable the hunk path.  Falls back to ``processing_class`` if
            ``tokenizer`` is ``None`` and ``processing_class`` is provided.

    Returns:
        A configured :class:`DiffAwareSFTTrainer` instance.

    Raises:
        RuntimeError: If ``resolved_tokenizer.pad_token_id`` is ``None``.
            trl's DataCollatorForLanguageModeling requires an explicit pad id.
    """
    # trl 0.19+ relocated DataCollatorForLanguageModeling (with completion-only
    # masking) into trl.trainer.sft_trainer; the legacy top-level
    # DataCollatorForCompletionOnlyLM was removed.
    from trl.trainer.sft_trainer import (  # type: ignore[attr-defined]
        DataCollatorForLanguageModeling,
    )

    resolved_tokenizer = tokenizer if tokenizer is not None else processing_class

    pad_token_id = getattr(resolved_tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(resolved_tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise RuntimeError(
            "build_diff_aware_sft_trainer requires a tokenizer with "
            "pad_token_id or eos_token_id set."
        )

    # Forward args.max_length / args.truncation_mode so the diff-aware path
    # respects the SFTConfig sequence cap. The standard SFTTrainer path threads
    # these into the collator at sft_trainer.py:893; bypassing that path (as we
    # do here) without forwarding the cap means every sequence reaches the GPU
    # uncapped — at Qwen3.5-9B vocab=152k, the cross-entropy logits tensor
    # alone OOMs an L4 at seq_len ~8k.
    inner_collator = DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        completion_only_loss=True,
        max_length=getattr(args, "max_length", None),
        truncation_mode=getattr(args, "truncation_mode", "keep_start"),
    )
    collator = DiffWeightedDataCollator(
        inner_collator,
        changed_weight=changed_weight,
        unchanged_weight=unchanged_weight,
        tokenizer=resolved_tokenizer,
    )

    trainer = DiffAwareSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processing_class,
        data_collator=collator,
    )
    # Mirror the collator's weights onto the trainer so per-step metrics
    # split changed vs context using the same midpoint the collator used.
    trainer._diff_changed_weight = changed_weight
    trainer._diff_unchanged_weight = unchanged_weight
    return trainer
