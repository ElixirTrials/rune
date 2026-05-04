"""Ingest external HuggingFace code-review data into the D2L pair schema.

Converts rows from ``ronantakizawa/github-codereview`` into the same
``activation_text`` / ``teacher_text`` pair dicts that
:func:`~model_training.d2l_data.unroll_trajectory_to_pairs` emits.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "codereview_row_to_pair",
    "ingest_codereview_to_pairs",
    "load_codereview_dataset",
]

_HF_DATASET = "ronantakizawa/github-codereview"


def load_codereview_dataset(
    split: str = "train",
    max_rows: int | None = None,
    streaming: bool = False,
) -> Any:
    """Load the ``ronantakizawa/github-codereview`` HuggingFace dataset.

    Args:
        split: Dataset split to load (e.g. ``"train"``).
        max_rows: If set, return only the first ``max_rows`` rows via
            ``dataset.select(range(max_rows))``.  Ignored when
            ``streaming=True``.
        streaming: Pass ``True`` to return an ``IterableDataset`` instead
            of loading the full split into memory.

    Returns:
        A HuggingFace ``Dataset`` or ``IterableDataset`` object.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(_HF_DATASET, split=split, streaming=streaming)
    if not streaming and max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))  # type: ignore[arg-type]
    return ds


def codereview_row_to_pair(
    row: dict[str, Any],
    quality_config: Any | None = None,
) -> dict[str, Any] | None:
    """Convert one ``ronantakizawa/github-codereview`` row to a pair dict.

    Rows are rejected (``None`` returned) when:

    * ``before_code``, ``after_code``, or ``reviewer_comment`` is missing
      or empty.
    * ``is_negative`` is ``True``.

    The returned dict matches the schema produced by
    :func:`~model_training.d2l_data.unroll_trajectory_to_pairs` so it is
    compatible with :func:`~model_training.d2l_data.pairs_to_chat_messages`
    and :class:`~model_training.diff_loss.DiffWeightedDataCollator`.

    Args:
        row: A single row dict from the dataset.
        quality_config: Optional
            :class:`~model_training.d2l_quality.QualityWeightConfig`
            instance.  Uses defaults when ``None``.

    Returns:
        Pair dict, or ``None`` for degenerate / negative rows.
    """
    from model_training.d2l_quality import (  # noqa: PLC0415
        QualityWeightConfig,
        score_external_quality,
    )

    # --- filter degenerate / negative rows ---
    if row.get("is_negative"):
        return None

    before_code: str = row.get("before_code") or ""
    after_code: str = row.get("after_code") or ""
    reviewer_comment: str = row.get("reviewer_comment") or ""

    if not before_code.strip() or not after_code.strip():
        return None
    if not reviewer_comment.strip():
        return None

    repo_name: str = row.get("repo_name") or "unknown"
    pr_number: str | int = row.get("pr_number") or "unknown"
    file_path: str = row.get("file_path") or "unknown"

    q_cfg = (
        quality_config
        if isinstance(quality_config, QualityWeightConfig)
        else QualityWeightConfig()
    )
    quality_score = score_external_quality(
        feedback_body=reviewer_comment,
        before_code=before_code,
        after_code=after_code,
        config=q_cfg,
    )

    task_description = (
        f"Review and revise code from {repo_name} "
        f"(PR #{pr_number}, file: {file_path})"
    )
    activation_text = (
        f"## Task\n{task_description}"
        f"\n\n## Current Code\n{before_code}"
        f"\n\n## Review Feedback\n{reviewer_comment}"
    )
    teacher_text = f"{activation_text}\n\n## Revision\n{after_code}"

    task_id = f"codereview_{repo_name}_{pr_number}_{file_path}"
    source_task_id = f"codereview_{repo_name}_{pr_number}"

    return {
        "task_id": task_id,
        "activation_text": activation_text,
        "teacher_text": teacher_text,
        "pre_code": before_code,
        "post_code": after_code,
        "quality_score": quality_score,
        "metadata": {
            "source": "external_codereview",
            "source_type": "external_single_turn",
            "source_task_id": source_task_id,
            "step_index": 0,
            "quality_score": quality_score,
        },
    }


def ingest_codereview_to_pairs(
    split: str = "train",
    max_rows: int | None = None,
    quality_config: Any | None = None,
    min_quality_score: float = 0.0,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Ingest the code-review dataset and return a list of pair dicts.

    Loads ``ronantakizawa/github-codereview``, converts each row via
    :func:`codereview_row_to_pair`, and filters out ``None`` results and
    rows whose ``quality_score`` is below ``min_quality_score``.

    Args:
        split: Dataset split to load.
        max_rows: Cap on total rows to read (applied before conversion).
        quality_config: Optional
            :class:`~model_training.d2l_quality.QualityWeightConfig`
            instance forwarded to :func:`codereview_row_to_pair`.
        min_quality_score: Pairs with ``quality_score < min_quality_score``
            are discarded.
        streaming: Load dataset in streaming mode.

    Returns:
        List of pair dicts ready for
        :func:`~model_training.d2l_data.pairs_to_chat_messages`.
    """
    ds = load_codereview_dataset(
        split=split, max_rows=max_rows, streaming=streaming
    )

    pairs: list[dict[str, Any]] = []
    for row in ds:
        pair = codereview_row_to_pair(row, quality_config=quality_config)
        if pair is None:
            continue
        if pair["quality_score"] < min_quality_score:
            continue
        pairs.append(pair)
    return pairs
