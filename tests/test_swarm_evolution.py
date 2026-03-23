"""Tests for scripts/swarm_evolution.py — evolution sweep and evaluation."""

import sys
from pathlib import Path
from unittest.mock import patch

from adapter_registry import AdapterRegistry
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from swarm_evolution import (
    MERGE_MIN_ADAPTERS,
    PRUNE_FITNESS_THRESHOLD,
    evaluate_adapter,
    evolution_sweep,
)


def _make_registry():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return AdapterRegistry(engine=engine)


def _make_record(registry, adapter_id, task_type="bug-fix", fitness=None, **kwargs):
    from adapter_registry.models import AdapterRecord

    record = AdapterRecord(
        id=adapter_id,
        version=1,
        task_type=task_type,
        base_model_id="Qwen/Qwen2.5-Coder-7B",
        rank=16,
        created_at="2026-01-01T00:00:00Z",
        file_path=f"/adapters/{adapter_id}.safetensors",
        file_hash="abc123",
        file_size_bytes=1024,
        pass_rate=fitness,
        fitness_score=fitness,
        source="distillation",
        session_id="test-session",
        **kwargs,
    )
    registry.store(record)
    return record


def test_evaluate_adapter_updates_registry() -> None:
    registry = _make_registry()
    _make_record(registry, "a1")
    fitness = evaluate_adapter(
        "a1", pass_rate=0.8, diversity_score=0.2, registry=registry
    )
    record = registry.retrieve_by_id("a1")
    assert record.pass_rate == 0.8
    assert record.fitness_score == fitness
    assert fitness > 0


def test_evolution_sweep_prunes_low_fitness() -> None:
    registry = _make_registry()
    _make_record(registry, "good", fitness=0.8)
    _make_record(registry, "bad", fitness=0.1)
    evolution_sweep(registry)
    bad_record = registry.retrieve_by_id("bad")
    assert bad_record.is_archived is True
    good_record = registry.retrieve_by_id("good")
    assert good_record.is_archived is False


def test_evolution_sweep_skips_when_few_adapters() -> None:
    registry = _make_registry()
    for i in range(MERGE_MIN_ADAPTERS - 1):
        _make_record(registry, f"a{i}", fitness=0.8)
    with patch("swarm_evolution._ties_merge_adapters") as mock_merge:
        evolution_sweep(registry)
        mock_merge.assert_not_called()


def test_evolution_sweep_merges_when_enough_adapters() -> None:
    registry = _make_registry()
    for i in range(MERGE_MIN_ADAPTERS):
        _make_record(registry, f"a{i}", fitness=0.5 + i * 0.05)
    with patch("swarm_evolution._ties_merge_adapters") as mock_merge:
        mock_merge.return_value = "merged-id"
        summary = evolution_sweep(registry)
        mock_merge.assert_called_once()
        assert summary["task_types"]["bug-fix"]["merged"] == 1


def test_evolution_sweep_no_prune_above_threshold() -> None:
    registry = _make_registry()
    _make_record(registry, "a1", fitness=PRUNE_FITNESS_THRESHOLD + 0.1)
    evolution_sweep(registry)
    record = registry.retrieve_by_id("a1")
    assert record.is_archived is False


def test_evolution_sweep_excludes_null_fitness_from_merge() -> None:
    """Adapters with NULL fitness should not be selected for merging."""
    registry = _make_registry()
    # Create enough adapters to trigger merge — some with NULL fitness
    for i in range(MERGE_MIN_ADAPTERS):
        if i < 2:
            _make_record(registry, f"a{i}", fitness=None)
        else:
            _make_record(registry, f"a{i}", fitness=0.5 + i * 0.05)
    with patch("swarm_evolution._ties_merge_adapters") as mock_merge:
        mock_merge.return_value = "merged-id"
        evolution_sweep(registry)
        if mock_merge.called:
            parent_ids = mock_merge.call_args[0][0]
            for aid in parent_ids:
                record = registry.retrieve_by_id(aid)
                assert record.fitness_score is not None
