"""Unit tests for oracle_cache module (CPU-only)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from model_training.oracle_cache import (
    DIAGNOSE_BIN_KEY,
    ORACLE_ID_PREFIX,
    OracleAdapterCache,
    _bin_key_for_record,
    audit_oracle_coverage,
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
    record: dict[str, object],
    expected: str,
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


def test_audit_oracle_coverage_full_coverage() -> None:
    """Every record has a registered oracle → ratio 1.0."""
    registry = MagicMock()
    registry.retrieve_by_id.return_value = _fake_record(
        adapter_id="irrelevant", file_path="/any/path"
    )

    records = [
        {"metadata": {"phase": "decompose", "benchmark": "humaneval"}},
        {"metadata": {"phase": "plan", "benchmark": "mbpp"}},
        {"metadata": {"phase": "diagnose", "benchmark": "apps"}},
    ]
    ratio, counts = audit_oracle_coverage(records, registry)

    assert ratio == pytest.approx(1.0)
    assert counts == {
        "decompose_humaneval": 1,
        "plan_mbpp": 1,
        "diagnose_pooled": 1,
    }


def test_audit_oracle_coverage_partial() -> None:
    """Unregistered bins subtract from the coverage ratio."""
    from adapter_registry.exceptions import AdapterNotFoundError

    registry = MagicMock()

    def _fake_lookup(adapter_id: str) -> MagicMock:
        if adapter_id == "oracle_plan_mbpp":
            raise AdapterNotFoundError("missing")
        return _fake_record(adapter_id=adapter_id, file_path=f"/a/{adapter_id}")

    registry.retrieve_by_id.side_effect = _fake_lookup

    records = [
        {"metadata": {"phase": "decompose", "benchmark": "humaneval"}},
        {"metadata": {"phase": "plan", "benchmark": "mbpp"}},  # missing
        {"metadata": {"phase": "plan", "benchmark": "mbpp"}},  # missing (dup bin)
        {"metadata": {"phase": "code", "benchmark": "apps"}},
    ]
    ratio, counts = audit_oracle_coverage(records, registry)

    assert ratio == pytest.approx(0.5)  # 2 of 4 records covered
    assert counts["plan_mbpp"] == 2
    assert counts["decompose_humaneval"] == 1


def test_audit_oracle_coverage_empty_records() -> None:
    """Empty record list returns 0.0 coverage and empty counts."""
    registry = MagicMock()
    ratio, counts = audit_oracle_coverage([], registry)
    assert ratio == pytest.approx(0.0)
    assert counts == {}


def test_audit_oracle_coverage_skips_unroutable_records() -> None:
    """Unroutable records penalise coverage but are excluded from bin_counts.

    The denominator of coverage_ratio is the **original** len(records), so
    unroutable records (those raising ValueError from _bin_key_for_record)
    reduce the ratio. They do not appear in bin_counts because no bin could
    be derived for them.
    """
    registry = MagicMock()
    registry.retrieve_by_id.return_value = _fake_record(
        adapter_id="oracle_decompose_humaneval",
        file_path="/adapters/oracle_decompose_humaneval",
    )
    records = [
        {"metadata": {"phase": "decompose", "benchmark": "humaneval"}},  # covered
        {"task_id": "not-parseable"},  # unroutable
    ]
    ratio, counts = audit_oracle_coverage(records, registry)
    # 1 covered out of 2 total (unroutable counts against denominator)
    assert ratio == pytest.approx(0.5)
    assert counts == {"decompose_humaneval": 1}
    # The unroutable record never triggered a registry lookup.
    registry.retrieve_by_id.assert_called_once()


def _fake_lora_dict(marker: str) -> dict[str, dict[str, object]]:
    """Return a sentinel-shaped LoRA dict distinguishable by marker."""
    return {
        "q_proj": {
            "A": MagicMock(name=f"A:{marker}"),
            "B": MagicMock(name=f"B:{marker}"),
        }
    }


def test_oracle_cache_loads_once_per_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second .get() for the same bin returns the cached lora_dict (no reload)."""
    from model_training import oracle_cache

    load_calls: list[str] = []

    def _fake_loader(path: str, hc: object) -> dict[str, dict[str, object]]:
        load_calls.append(path)
        return _fake_lora_dict(path)

    monkeypatch.setattr(oracle_cache, "_load_oracle_as_lora_dict", _fake_loader)

    registry = MagicMock()
    registry.retrieve_by_id.return_value = _fake_record(
        adapter_id="oracle_decompose_humaneval",
        file_path="/a/oracle_decompose_humaneval",
    )

    cache = OracleAdapterCache(registry=registry, hc=MagicMock(), max_loaded=4)

    first = cache.get("decompose_humaneval")
    second = cache.get("decompose_humaneval")

    assert first is second
    assert load_calls == ["/a/oracle_decompose_humaneval"]


def test_oracle_cache_returns_none_when_bin_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unregistered bins return None; the loader is not called."""
    from adapter_registry.exceptions import AdapterNotFoundError
    from model_training import oracle_cache

    called = []
    monkeypatch.setattr(
        oracle_cache,
        "_load_oracle_as_lora_dict",
        lambda *a, **kw: called.append(a) or _fake_lora_dict("nope"),
    )

    registry = MagicMock()
    registry.retrieve_by_id.side_effect = AdapterNotFoundError("missing")

    cache = OracleAdapterCache(registry=registry, hc=MagicMock(), max_loaded=4)
    assert cache.get("plan_mbpp") is None
    assert called == []

    # Second lookup should hit the _missing set, not the registry.
    assert cache.get("plan_mbpp") is None
    registry.retrieve_by_id.assert_called_once()
    assert called == []


def test_oracle_cache_evicts_lru_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filling past max_loaded evicts the least-recently-used bin."""
    from model_training import oracle_cache

    load_calls: list[str] = []
    monkeypatch.setattr(
        oracle_cache,
        "_load_oracle_as_lora_dict",
        lambda path, hc: load_calls.append(path) or _fake_lora_dict(path),
    )

    registry = MagicMock()
    registry.retrieve_by_id.side_effect = lambda aid: _fake_record(
        adapter_id=aid, file_path=f"/a/{aid}"
    )

    cache = OracleAdapterCache(registry=registry, hc=MagicMock(), max_loaded=2)
    cache.get("plan_mbpp")  # LRU = [plan_mbpp]
    cache.get("code_humaneval")  # LRU = [plan_mbpp, code_humaneval]
    cache.get("plan_mbpp")  # LRU = [code_humaneval, plan_mbpp]
    cache.get("integrate_apps")  # evicts code_humaneval

    before = len(load_calls)
    cache.get("code_humaneval")  # re-load after eviction
    after = len(load_calls)
    assert after == before + 1


def test_oracle_cache_clear_releases_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """clear() empties the cache; subsequent get() reloads."""
    from model_training import oracle_cache

    load_calls: list[str] = []
    monkeypatch.setattr(
        oracle_cache,
        "_load_oracle_as_lora_dict",
        lambda path, hc: load_calls.append(path) or _fake_lora_dict(path),
    )

    registry = MagicMock()
    registry.retrieve_by_id.side_effect = lambda aid: _fake_record(
        adapter_id=aid, file_path=f"/a/{aid}"
    )

    cache = OracleAdapterCache(registry=registry, hc=MagicMock(), max_loaded=4)
    cache.get("plan_mbpp")
    cache.clear()
    cache.get("plan_mbpp")
    assert len(load_calls) == 2


def test_oracle_cache_rejects_non_positive_max_loaded() -> None:
    """max_loaded must be >= 1."""
    registry = MagicMock()
    with pytest.raises(ValueError, match="max_loaded must be >= 1"):
        OracleAdapterCache(registry=registry, hc=MagicMock(), max_loaded=0)
    with pytest.raises(ValueError, match="max_loaded must be >= 1"):
        OracleAdapterCache(registry=registry, hc=MagicMock(), max_loaded=-5)
