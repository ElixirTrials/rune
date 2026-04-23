"""Unit tests for oracle_cache module (CPU-only)."""

from __future__ import annotations

import pytest
from model_training.oracle_cache import (
    DIAGNOSE_BIN_KEY,
    ORACLE_ID_PREFIX,
    _bin_key_for_record,
)


def test_bin_key_from_metadata_phase_and_benchmark() -> None:
    """Bin key = '<phase>_<benchmark>' when both metadata fields are present."""
    record = {
        "task_id": "humaneval/HumanEval/0/decompose",
        "metadata": {"phase": "decompose", "benchmark": "humaneval"},
    }
    assert _bin_key_for_record(record) == "decompose_humaneval"


def test_bin_key_diagnose_is_pooled() -> None:
    """Diagnose bin is pooled across benchmarks regardless of metadata.benchmark."""
    record = {
        "task_id": "mbpp/MBPP/7/diagnose",
        "metadata": {"phase": "diagnose", "benchmark": "mbpp"},
    }
    assert _bin_key_for_record(record) == DIAGNOSE_BIN_KEY
    assert DIAGNOSE_BIN_KEY == "diagnose_pooled"


def test_bin_key_fallback_parses_task_id() -> None:
    """When metadata is missing, parse task_id of form '<benchmark>/<pid>/<phase>'."""
    record = {"task_id": "bigcodebench/BCB-42/plan"}
    assert _bin_key_for_record(record) == "plan_bigcodebench"


def test_bin_key_raises_on_unresolvable_record() -> None:
    """Records with neither metadata nor parseable task_id raise ValueError."""
    with pytest.raises(ValueError, match="cannot derive bin_key"):
        _bin_key_for_record({"task_id": "not-a-valid-task-id"})


def test_bin_key_metadata_overrides_task_id() -> None:
    """When both are present, metadata wins (more authoritative source)."""
    record = {
        "task_id": "humaneval/HE-1/code",
        "metadata": {"phase": "integrate", "benchmark": "mbpp"},
    }
    assert _bin_key_for_record(record) == "integrate_mbpp"


def test_oracle_id_prefix_constant() -> None:
    """ORACLE_ID_PREFIX matches trainer_bridge.py's adapter_id scheme."""
    assert ORACLE_ID_PREFIX == "oracle_"
