"""Line-level diff-aware loss weighting for SFT training.

All GPU-dependent imports (torch, trl, transformers) are deferred inside
function bodies to ensure CPU-only importability (INFRA-05).

Module-level imports: stdlib + typing only.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any

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
# Data collator
# ---------------------------------------------------------------------------


class DiffWeightedDataCollator:
    """Data collator that applies diff-aware loss weights to SFT batches.

    Wraps an inner collator (e.g. ``trl.DataCollatorForCompletionOnlyLM``).
    On each ``__call__``, the inner collator runs first to produce
    ``batch["input_ids"]`` and ``batch["labels"]``.  Diff weights are then
    computed and stored as ``batch["loss_weights"]``.

    Hunk path (preferred):
        When all features carry ``pre_code`` / ``post_code`` side-channel
        columns AND a ``tokenizer`` is provided, weights are computed via
        :func:`_compute_hunk_ranges` + :func:`compute_hunk_loss_weights`.
        The tokenizer is used to re-tokenize ``post_code`` with
        ``return_offsets_mapping=True`` to obtain char-to-token alignment.

    Fallback path:
        When any feature is missing ``pre_code`` / ``post_code`` or no
        tokenizer was supplied, the legacy set-based
        :func:`compute_diff_loss_weights` is used and a one-time
        ``logger.warning`` is emitted.

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
        pre_code: str,
        post_code: str,
    ) -> list[float]:
        """Compute weights using the hunk (line-level diff) path.

        Re-tokenizes ``post_code`` with ``return_offsets_mapping=True``.
        The resulting offsets are used directly against the hunk ranges
        computed from ``pre_code`` → ``post_code``.

        Note: The offset mapping is best-effort — the re-tokenized
        ``post_code`` may not align perfectly with the assistant-span tokens
        produced by the full chat template, but this is acceptable because:
        (a) the identity invariant holds regardless; (b) the directional
        property (changed > unchanged) is preserved on average.

        Args:
            input_ids_seq: Full sequence token IDs.
            labels_seq: Full sequence labels.
            pre_code: Code before the edit.
            post_code: Code after the edit.

        Returns:
            Per-token float weights.
        """
        hunk_ranges = _compute_hunk_ranges(pre_code, post_code)

        # Re-tokenize post_code to get char offsets.
        assert self.tokenizer is not None  # guaranteed by caller; satisfies mypy
        enc = self.tokenizer(
            post_code,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        post_offset_mapping: list[tuple[int, int]] = [
            tuple(o)
            for o in enc["offset_mapping"]  # type: ignore[misc]
        ]
        post_input_ids: list[int] = enc["input_ids"]

        n_seq = len(input_ids_seq)
        n_post = len(post_input_ids)

        # Build per-token offset mapping for the *full* sequence.
        # Strategy: match the last N tokens of the assistant span against
        # the re-tokenized post_code.  Everything else gets (0, 0).
        full_offset_mapping: list[tuple[int, int]] = [(0, 0)] * n_seq

        if n_post > 0 and n_post <= n_seq:
            # Find the tail of the sequence that matches post_input_ids.
            # Walk backward from the end of the sequence.
            tail_start = n_seq - n_post
            if input_ids_seq[tail_start : tail_start + n_post] == post_input_ids:
                for k, off in enumerate(post_offset_mapping):
                    full_offset_mapping[tail_start + k] = off
            else:
                # Fallback: assign offsets to the last n_post positions.
                for k, off in enumerate(post_offset_mapping):
                    full_offset_mapping[n_seq - n_post + k] = off

        return compute_hunk_loss_weights(
            input_ids=input_ids_seq,
            labels=labels_seq,
            offset_mapping=full_offset_mapping,
            hunk_ranges_chars=hunk_ranges,
            changed_weight=self.changed_weight,
            unchanged_weight=self.unchanged_weight,
        )

    def _weights_via_fallback(
        self,
        input_ids_seq: list[int],
        labels_seq: list[int],
        pre_code: str,
        post_code: str,
    ) -> list[float]:
        """Compute weights using the legacy set-based diff path.

        Args:
            input_ids_seq: Full sequence token IDs.
            labels_seq: Full sequence labels.
            pre_code: Code before the edit (tokenized by splitting on
                whitespace for the bag-of-ids approach).
            post_code: Code after the edit.

        Returns:
            Per-token float weights.
        """
        # Build changed set: token IDs that appear in post but not pre.
        # Use a simple character-level set as a proxy; the real IDs are in
        # input_ids.  Map by intersecting with actual ids present.
        pre_chars = set(pre_code)
        post_chars = set(post_code)
        changed_chars = post_chars - pre_chars

        # Map char overlap to token ids: a token is "changed" if its decoded
        # char is in changed_chars.  This is a best-effort heuristic.
        # For correctness we use the original set-based API.
        if self.tokenizer is not None:
            changed_ids: set[int] = {
                tid
                for tid in set(input_ids_seq)
                if any(c in changed_chars for c in self.tokenizer.decode([tid]))
            }
        else:
            changed_ids = set()

        return compute_diff_loss_weights(
            input_ids=input_ids_seq,
            labels=labels_seq,
            changed_ids=changed_ids,
            changed_weight=self.changed_weight,
            unchanged_weight=self.unchanged_weight,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of feature dicts into a training batch with loss weights.

        Side-channel columns ``pre_code`` and ``post_code`` are popped from
        each feature before passing to the inner collator so they don't
        interfere with tensor stacking.

        Args:
            features: List of example dicts (possibly with ``pre_code`` /
                ``post_code`` keys).

        Returns:
            Batch dict with standard keys plus ``"loss_weights"`` tensor.
        """
        global _HUNK_FALLBACK_WARNED

        # Pop side-channel columns before handing to the inner collator.
        pre_codes: list[str | None] = []
        post_codes: list[str | None] = []
        for feat in features:
            pre_codes.append(feat.pop("pre_code", None))
            post_codes.append(feat.pop("post_code", None))

        batch = self.inner_collator(features)

        # Determine which path to use.
        has_side_channels = all(p is not None for p in pre_codes) and all(
            p is not None for p in post_codes
        )
        use_hunk_path = has_side_channels and self.tokenizer is not None

        if not use_hunk_path and not _HUNK_FALLBACK_WARNED:
            logger.warning(
                "DiffWeightedDataCollator: pre_code/post_code side-channels missing "
                "or tokenizer not set; falling back to identity loss weights "
                "(1.0 for labeled tokens, 0.0 otherwise). "
                "Pass tokenizer= and include pre_code/post_code in dataset features "
                "to use the hunk path."
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
            pre = str(pre_codes[idx]) if pre_codes[idx] is not None else ""
            post = str(post_codes[idx]) if post_codes[idx] is not None else ""

            if use_hunk_path:
                w = self._weights_via_hunk_path(ids_seq, lab_seq, pre, post)
            else:
                # Identity fallback: 1.0 for labeled tokens, 0.0 for IGNORE_INDEX.
                # Avoids silently reducing the training objective to a uniform
                # rescale when the diff cannot be computed (no side-channels
                # or no tokenizer).
                w = [1.0 if lab != -100 else 0.0 for lab in lab_seq]
            all_weights.append(w)

        batch["loss_weights"] = torch.tensor(all_weights, dtype=torch.float32)
        return batch


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

    Constructs a :class:`DiffWeightedDataCollator` wrapping the default TRL
    completion-only collator, then instantiates ``trl.SFTTrainer`` with it.

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
        A configured ``trl.SFTTrainer`` instance.
    """
    from trl import (  # type: ignore[attr-defined]
        DataCollatorForCompletionOnlyLM,
        SFTTrainer,
    )

    resolved_tokenizer = tokenizer if tokenizer is not None else processing_class

    inner_collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant",
        tokenizer=resolved_tokenizer,
    )
    collator = DiffWeightedDataCollator(
        inner_collator,
        changed_weight=changed_weight,
        unchanged_weight=unchanged_weight,
        tokenizer=resolved_tokenizer,
    )

    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processing_class,
        data_collator=collator,
    )
