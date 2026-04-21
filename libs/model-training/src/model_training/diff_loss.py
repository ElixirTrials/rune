"""Diff-aware loss weighting for procedural trajectory SFT.

Mined GitHub pair records follow a review→revision structure: each
``activation_text`` contains the task, current code, and reviewer feedback;
each ``teacher_text`` appends the revised code. When we train on pairs with
``assistant_only_loss=True``, the assistant response often reproduces
substantial unchanged context (project setup, imports, unaltered lines)
before emitting the actual *delta* introduced by the revision. Uniform
loss weighting over those tokens dilutes the procedural signal we care
about — the correction itself.

This module implements a lightweight diff-aware loss weighter:

* ``compute_diff_loss_weights`` is a pure function — given per-token
  ``input_ids`` and ``labels`` (with user turns masked to ``-100`` by
  TRL's ``assistant_only_loss`` logic), it produces per-token float
  weights that are higher for assistant tokens whose token id does NOT
  appear anywhere in the user/context span, and lower for assistant
  tokens that match context tokens. Masked positions receive ``0.0``.

* ``DiffWeightedDataCollator`` wraps an inner collator (TRL's default
  assistant-only SFT collator), computes ``loss_weights`` from the
  resulting ``input_ids`` / ``labels`` via ``compute_diff_loss_weights``,
  and attaches them to the batch dict. All heavy imports (``torch``,
  ``trl``) are deferred so this module is CPU-safe.

* ``DiffAwareSFTTrainer`` subclasses ``trl.SFTTrainer`` to multiply the
  per-token CE loss by the collator's ``loss_weights`` before averaging.
  With uniform weights (``changed == unchanged == 1.0``) it is
  identity-equivalent to vanilla ``SFTTrainer``; the test suite verifies
  this regression guard.

Enabled by passing ``diff_aware_loss=True`` to
``model_training.trainer.train_qlora``. Otherwise the existing
``SFTTrainer`` + ``assistant_only_loss=True`` path runs unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "compute_diff_loss_weights",
    "DiffWeightedDataCollator",
    "build_diff_aware_sft_trainer",
]

# Sentinel used by HF/TRL for "ignore in loss". Duplicated here so this
# module is CPU-importable without pulling in transformers.
IGNORE_INDEX = -100


def compute_diff_loss_weights(
    input_ids: list[int],
    labels: list[int],
    *,
    changed_weight: float = 1.0,
    unchanged_weight: float = 0.3,
) -> list[float]:
    """Return per-token loss weights biased toward the assistant's delta.

    The context-token set is every ``input_ids[i]`` where
    ``labels[i] == IGNORE_INDEX`` — i.e. every token position that the
    SFT collator already masked (system + user turns when
    ``assistant_only_loss=True``). Assistant-turn tokens whose token id
    appears in that set are presumed boilerplate / carried-over context
    and get ``unchanged_weight``; tokens whose id is NEW (not present
    anywhere in the masked context) are presumed to be the revision
    itself and get ``changed_weight``. Masked positions receive ``0.0``.

    Identity guarantee: when ``changed_weight == unchanged_weight``, every
    non-masked position receives that single weight, so scaling the loss
    by the weight tensor is equivalent to multiplying by a constant —
    gradient direction is unchanged vs. unweighted SFT.

    Args:
        input_ids: Flat list of token ids for one sequence.
        labels: Same-length list; ``IGNORE_INDEX`` marks masked positions.
        changed_weight: Weight applied to assistant tokens whose id is not
            present in the masked context.
        unchanged_weight: Weight applied to assistant tokens whose id is
            already present in the masked context.

    Returns:
        A list of floats the same length as ``input_ids``.

    Raises:
        ValueError: If ``input_ids`` and ``labels`` have different lengths.
    """
    if len(input_ids) != len(labels):
        raise ValueError(
            f"input_ids and labels must have equal length, "
            f"got {len(input_ids)} and {len(labels)}"
        )
    context_token_ids = {
        tid for tid, lbl in zip(input_ids, labels, strict=True) if lbl == IGNORE_INDEX
    }
    weights: list[float] = []
    for tid, lbl in zip(input_ids, labels, strict=True):
        if lbl == IGNORE_INDEX:
            weights.append(0.0)
        elif tid in context_token_ids:
            weights.append(unchanged_weight)
        else:
            weights.append(changed_weight)
    return weights


class DiffWeightedDataCollator:
    """Collator wrapper that augments batches with ``loss_weights``.

    Delegates input_ids / attention_mask / labels construction to an
    inner collator (TRL's default assistant-only SFT collator) and adds
    a ``loss_weights`` float tensor computed by
    :func:`compute_diff_loss_weights` on each sequence independently.

    Used together with :class:`DiffAwareSFTTrainer`. ``torch`` is imported
    lazily inside ``__call__`` so instantiation works in CPU-only tests.
    """

    def __init__(
        self,
        inner_collator: Any,
        *,
        changed_weight: float = 1.0,
        unchanged_weight: float = 0.3,
    ) -> None:
        """Wrap ``inner_collator`` and remember the diff weights.

        Args:
            inner_collator: Any callable producing a batch dict with
                ``input_ids`` and ``labels`` tensors (the TRL default
                SFT collator is the intended choice).
            changed_weight: Per-token weight for assistant tokens whose id
                is not present in the masked context span.
            unchanged_weight: Per-token weight for assistant tokens whose
                id is already in the masked context span.
        """
        self.inner = inner_collator
        self.changed_weight = changed_weight
        self.unchanged_weight = unchanged_weight

    def __call__(self, features: list[Any]) -> dict[str, Any]:
        """Produce a batch dict with an extra ``loss_weights`` float tensor."""
        import torch  # noqa: PLC0415

        batch = self.inner(features)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        weight_rows: list[list[float]] = []
        for row_ids, row_lbl in zip(
            input_ids.tolist(), labels.tolist(), strict=True
        ):
            weight_rows.append(
                compute_diff_loss_weights(
                    row_ids,
                    row_lbl,
                    changed_weight=self.changed_weight,
                    unchanged_weight=self.unchanged_weight,
                )
            )
        batch["loss_weights"] = torch.tensor(
            weight_rows, dtype=torch.float32, device=input_ids.device
        )
        return batch


def build_diff_aware_sft_trainer(
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    processing_class: Any,
    peft_config: Any = None,
    changed_weight: float = 1.0,
    unchanged_weight: float = 0.3,
) -> Any:
    """Construct a ``DiffAwareSFTTrainer`` with deferred ``trl`` / ``torch`` imports.

    Factored out so :mod:`model_training.trainer` can obtain the trainer
    without importing ``trl`` at module top — and so tests can inspect
    the class for signature-level assertions without instantiating it.
    """
    import torch  # noqa: PLC0415
    from torch.nn import CrossEntropyLoss  # noqa: PLC0415
    from trl import SFTTrainer  # noqa: PLC0415

    class DiffAwareSFTTrainer(SFTTrainer):  # type: ignore[misc]
        """SFTTrainer variant that multiplies per-token CE by ``loss_weights``.

        Identity when loss_weights ≡ 1.0 over non-masked positions (tested).
        """

        def compute_loss(  # type: ignore[override]
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            weights = inputs.pop("loss_weights", None)
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="none")
            per_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.shape)
            if weights is not None:
                shift_w = weights[..., 1:].contiguous()
                per_tok = per_tok * shift_w
            mask = (shift_labels != IGNORE_INDEX).to(per_tok.dtype)
            denom = mask.sum().clamp_min(torch.tensor(1.0, device=mask.device))
            loss = (per_tok * mask).sum() / denom
            return (loss, outputs) if return_outputs else loss

    trainer = DiffAwareSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    return trainer
