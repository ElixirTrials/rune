"""Unit tests for oracle_cache module (CPU-only)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from model_training.oracle_cache import (
    DIAGNOSE_BIN_KEY,
    ORACLE_ID_PREFIX,
    _bin_key_for_record,
    lookup_oracle_path,
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


@pytest.mark.parametrize(
    "record,expected",
    [
        # phase in metadata, benchmark from task_id
        (
            {"task_id": "humaneval/HE-1/code", "metadata": {"phase": "plan"}},
            "plan_humaneval",
        ),
        # benchmark in metadata, phase from task_id
        (
            {"task_id": "mbpp/BCB-3/integrate", "metadata": {"benchmark": "mbpp"}},
            "integrate_mbpp",
        ),
    ],
)
def test_bin_key_partial_metadata_fills_from_task_id(
    record: dict[str, object], expected: str,
) -> None:
    """Partial metadata is supplemented by task_id parsing.

    Exercises the branch where only one of ``metadata.phase`` /
    ``metadata.benchmark`` is supplied; the missing field is filled from
    ``task_id`` parts.
    """
    assert _bin_key_for_record(record) == expected


def test_oracle_id_prefix_constant() -> None:
    """ORACLE_ID_PREFIX matches trainer_bridge.py's adapter_id scheme."""
    assert ORACLE_ID_PREFIX == "oracle_"


def _fake_record(
    adapter_id: str,
    file_path: str,
    is_archived: bool = False,
) -> MagicMock:
    """Build a fake AdapterRecord with the fields lookup_oracle_path reads."""
    rec = MagicMock()
    rec.id = adapter_id
    rec.file_path = file_path
    rec.is_archived = is_archived
    return rec


def test_lookup_oracle_path_returns_file_path() -> None:
    """Returns the registered file_path when the oracle exists."""
    registry = MagicMock()
    registry.retrieve_by_id.return_value = _fake_record(
        adapter_id="oracle_decompose_humaneval",
        file_path="/adapters/oracle_decompose_humaneval",
    )
    assert (
        lookup_oracle_path("decompose_humaneval", registry)
        == "/adapters/oracle_decompose_humaneval"
    )
    registry.retrieve_by_id.assert_called_once_with("oracle_decompose_humaneval")


def test_lookup_oracle_path_returns_none_when_missing() -> None:
    """Returns None when the registry has no record for the bin."""
    from adapter_registry.exceptions import AdapterNotFoundError

    registry = MagicMock()
    registry.retrieve_by_id.side_effect = AdapterNotFoundError(
        "oracle_plan_mbpp not found"
    )
    assert lookup_oracle_path("plan_mbpp", registry) is None


def test_lookup_oracle_path_returns_none_when_archived() -> None:
    """Archived adapters are treated as missing."""
    registry = MagicMock()
    registry.retrieve_by_id.return_value = _fake_record(
        adapter_id="oracle_code_apps",
        file_path="/adapters/oracle_code_apps",
        is_archived=True,
    )
    assert lookup_oracle_path("code_apps", registry) is None
