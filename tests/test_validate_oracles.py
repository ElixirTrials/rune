"""Tests for scripts/validate_oracles.py oracle validation runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "validate_oracles",
    Path(__file__).resolve().parent.parent / "scripts" / "validate_oracles.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_parse_oracle_spec_phase_benchmark() -> None:
    """'<phase>_<benchmark>:<adapter>' parses to (bin_key, benchmark, adapter)."""
    assert _MOD._parse_oracle_spec("decompose_humaneval:adapter-1") == (
        "decompose_humaneval",
        "humaneval",
        "adapter-1",
    )


def test_parse_oracle_spec_diagnose_pooled() -> None:
    """diagnose_pooled uses the pooled default benchmark."""
    bin_key, benchmark, adapter = _MOD._parse_oracle_spec("diagnose_pooled:adapter-99")
    assert bin_key == "diagnose_pooled"
    assert benchmark == _MOD.DEFAULT_POOLED_BENCHMARK
    assert adapter == "adapter-99"


def test_parse_oracle_spec_rejects_missing_colon() -> None:
    """Specs without ':' are rejected."""
    with pytest.raises(ValueError, match="<bin_key>:<adapter_id>"):
        _MOD._parse_oracle_spec("decompose_humaneval")


def test_parse_oracle_spec_rejects_bad_bin_key() -> None:
    """Bin keys that are neither <phase>_<benchmark> nor diagnose_pooled fail."""
    with pytest.raises(ValueError, match="<phase>_<benchmark>"):
        _MOD._parse_oracle_spec("justone:adapter-1")


def test_dry_run_produces_stub_results() -> None:
    """--dry-run emits one stub per spec without loading models."""
    results = _MOD.validate_oracles(
        base_model="Qwen/Qwen3.5-9B",
        oracle_specs=[
            "decompose_humaneval:adapter-a",
            "diagnose_pooled:adapter-b",
        ],
        dry_run=True,
    )
    assert len(results) == 2
    assert all(r["dry_run"] is True for r in results)
    assert results[0]["bin_key"] == "decompose_humaneval"
    assert results[1]["bin_key"] == "diagnose_pooled"


def test_validate_oracles_marks_pass_when_delta_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When stack beats base by >= threshold, oracle is marked passed."""

    def fake_eval(**_kwargs: object) -> dict[str, object]:
        return {
            "bin_key": "decompose_humaneval",
            "benchmark": "humaneval",
            "adapter_id": "adapter-x",
            "base_pass_at_1": 0.50,
            "stack_pass_at_1": 0.55,
            "delta": 0.05,
            "threshold": 0.03,
            "passed": True,
            "dry_run": False,
        }

    monkeypatch.setattr(_MOD, "_evaluate_one", fake_eval)
    results = _MOD.validate_oracles(
        base_model="Qwen/Qwen3.5-9B",
        oracle_specs=["decompose_humaneval:adapter-x"],
        threshold=0.03,
    )
    assert results[0]["passed"] is True
    assert results[0]["delta"] == 0.05
