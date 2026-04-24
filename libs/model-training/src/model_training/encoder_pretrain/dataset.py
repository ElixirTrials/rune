"""Contrastive pair dataset and tokenizing collator for encoder pretraining.

Loads augmented JSONL files from an output directory (produced by augment.py)
and exposes them as a PyTorch Dataset of (anchor, positive) string pairs.

ContrastiveCollator tokenizes both sides symmetrically and returns batched
input_ids / attention_mask tensors prefixed ``anchor_`` and ``positive_``.

All GPU-dependent imports (torch, transformers) are deferred inside class
bodies / method bodies per INFRA-05.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContrastivePairDataset:
    """PyTorch-compatible dataset of (encoder_input, post_code) pairs.

    Loads all ``*.jsonl`` files from a directory of augmented pairs. Rows
    where ``post_code`` is empty are skipped so the collator always receives
    a valid positive target.

    Args:
        rows: List of augmented pair dicts as produced by augment_corpus.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Initialise dataset from a list of augmented pair dicts.

        Args:
            rows: Augmented pair dicts; rows with empty post_code are skipped.
        """
        self._rows = [r for r in rows if r.get("post_code", "").strip()]
        skipped = len(rows) - len(self._rows)
        if skipped > 0:
            logger.warning(
                "ContrastivePairDataset: skipped %d rows with empty post_code",
                skipped,
            )

    @classmethod
    def from_dir(
        cls, augmented_dir: Path, glob: str = "*.jsonl"
    ) -> "ContrastivePairDataset":
        """Load all augmented JSONL files from augmented_dir.

        Args:
            augmented_dir: Directory containing augmented pair JSONL files.
            glob: Glob pattern to match files.

        Returns:
            ContrastivePairDataset with all rows combined.

        Raises:
            FileNotFoundError: If augmented_dir does not exist.
        """
        if not augmented_dir.exists():
            raise FileNotFoundError(
                f"augmented_dir does not exist: {augmented_dir}. "
                "Run augment_corpus first."
            )
        all_rows: list[dict[str, Any]] = []
        for path in sorted(augmented_dir.glob(glob)):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_rows.append(json.loads(line))
        logger.info(
            "ContrastivePairDataset.from_dir: loaded %d rows from %s",
            len(all_rows),
            augmented_dir,
        )
        return cls(all_rows)

    def __len__(self) -> int:  # noqa: D105
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, str]:  # noqa: D105
        row = self._rows[idx]
        return {
            "anchor": row["encoder_input"],
            "positive": row["post_code"],
        }


class ContrastiveCollator:
    """Tokenize (anchor, positive) string pairs into padded batch tensors.

    Returns a dict with keys ``anchor_input_ids``, ``anchor_attention_mask``,
    ``positive_input_ids``, ``positive_attention_mask`` — all
    ``LongTensor[batch, max_length]``.

    Args:
        tokenizer: A HuggingFace tokenizer (e.g. from AutoTokenizer).
        max_length: Maximum sequence length; sequences are truncated and
            padded to exactly this length.
    """

    def __init__(self, tokenizer: Any, max_length: int = 512) -> None:  # noqa: D107
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(self, items: list[dict[str, str]]) -> dict[str, Any]:
        """Tokenize a list of (anchor, positive) dicts into a batch.

        Args:
            items: List of dicts with ``anchor`` and ``positive`` string keys.

        Returns:
            Dict with ``anchor_input_ids``, ``anchor_attention_mask``,
            ``positive_input_ids``, ``positive_attention_mask`` tensors of
            shape ``(batch_size, max_length)``.
        """
        anchors = [item["anchor"] for item in items]
        positives = [item["positive"] for item in items]

        def _tokenize(texts: list[str]) -> dict[str, Any]:
            return self._tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )

        anchor_enc = _tokenize(anchors)
        positive_enc = _tokenize(positives)

        return {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
        }
