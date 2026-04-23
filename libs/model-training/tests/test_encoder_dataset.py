"""Tests for ContrastivePairDataset and ContrastiveCollator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _write_augmented_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _make_row(i: int) -> dict[str, Any]:
    return {
        "task_id": f"pr_{i:03d}",
        "pre_code": f"def foo_{i}(): pass",
        "post_code": f"def foo_{i}(): return {i}",
        "task_desc": f"Fix foo_{i} to return {i}",
        "task_desc_source": "explicit_field",
        "encoder_input": f"Fix foo_{i} to return {i}\n\ndef foo_{i}(): pass",
        "metadata": {},
    }


def test_dataset_len_and_getitem(tmp_path: Path) -> None:
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset

    rows = [_make_row(i) for i in range(5)]
    _write_augmented_jsonl(tmp_path / "repo.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    assert len(ds) == 5
    item = ds[0]
    assert "anchor" in item
    assert "positive" in item
    assert item["anchor"] == rows[0]["encoder_input"]
    assert item["positive"] == rows[0]["post_code"]


def test_dataset_from_dir_loads_multiple_files(tmp_path: Path) -> None:
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset

    for j in range(3):
        rows = [_make_row(j * 10 + i) for i in range(4)]
        _write_augmented_jsonl(tmp_path / f"repo_{j}.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    assert len(ds) == 12


def test_dataset_skips_rows_missing_post_code(tmp_path: Path) -> None:
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset

    rows = [_make_row(0), {**_make_row(1), "post_code": ""}, _make_row(2)]
    _write_augmented_jsonl(tmp_path / "repo.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    assert len(ds) == 2  # row with empty post_code skipped


def test_collator_returns_tokenized_batch(tmp_path: Path) -> None:
    """ContrastiveCollator tokenizes anchors and positives into input_ids."""
    from model_training.encoder_pretrain.dataset import (
        ContrastiveCollator,
        ContrastivePairDataset,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    rows = [_make_row(i) for i in range(4)]
    _write_augmented_jsonl(tmp_path / "repo.jsonl", rows)
    ds = ContrastivePairDataset.from_dir(tmp_path)
    collator = ContrastiveCollator(tokenizer=tokenizer, max_length=64)

    batch = collator([ds[i] for i in range(4)])
    assert "anchor_input_ids" in batch
    assert "anchor_attention_mask" in batch
    assert "positive_input_ids" in batch
    assert "positive_attention_mask" in batch
    assert batch["anchor_input_ids"].shape == (4, 64)
    assert batch["positive_input_ids"].shape == (4, 64)


def test_dataset_from_dir_missing_raises(tmp_path: Path) -> None:
    """from_dir raises FileNotFoundError for non-existent directory."""
    from model_training.encoder_pretrain.dataset import ContrastivePairDataset

    with pytest.raises(FileNotFoundError):
        ContrastivePairDataset.from_dir(tmp_path / "nonexistent")
