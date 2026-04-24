"""Tests for corpus_producer.trainer_bridge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from corpus_producer.trainer_bridge import _WARM_START, invoke_bin_training


def _write_manifest(tmpdir: str, n: int = 3) -> Path:
    p = Path(tmpdir) / "decompose_humaneval.jsonl"
    records = [
        {
            "task_id": f"humaneval/HE/{i}/decompose",
            "activation_text": "in",
            "teacher_text": "in\nout",
        }
        for i in range(n)
    ]
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def test_dry_run_returns_adapter_id_without_training():
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        adapter_id = invoke_bin_training("decompose_humaneval", mp, dry_run=True)
        assert adapter_id == "oracle_decompose_humaneval"


def test_adapter_id_format():
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        adapter_id = invoke_bin_training("diagnose_pooled", mp, dry_run=True)
        assert adapter_id == "oracle_diagnose_pooled"


def test_raises_if_manifest_missing():
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        invoke_bin_training(
            "decompose_humaneval", "/nonexistent/path.jsonl", dry_run=True
        )


@patch("corpus_producer.trainer_bridge.train_and_register")
def test_train_called_with_correct_args(mock_t: MagicMock) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        invoke_bin_training("decompose_humaneval", mp, dry_run=False)
        mock_t.assert_called_once()
        kwargs = mock_t.call_args[1]
        assert kwargs["adapter_id"] == "oracle_decompose_humaneval"
        assert kwargs["task_type"] == "decompose_humaneval"
        assert kwargs["warm_start_adapter_id"] == _WARM_START
        assert kwargs["rank"] == 64
        assert kwargs["diff_aware_loss"] is True


@patch("corpus_producer.trainer_bridge.train_and_register")
def test_train_called_with_dataset_path(mock_t: MagicMock) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp = _write_manifest(tmp)
        invoke_bin_training("plan_mbpp", mp, dry_run=False)
        kwargs = mock_t.call_args[1]
        assert kwargs["dataset_path"] == str(mp)
