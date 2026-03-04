"""Root conftest.py — shared factory fixtures for all Rune test suites.

This file lives at the repo root so pytest auto-discovers it for every test
under services/ and libs/. Each fixture returns a factory function that accepts
**kwargs for field overrides; unspecified fields use deterministic defaults.

Usage (no import needed in test files):

    def test_something(make_adapter_record):
        obj = make_adapter_record(task_type="code-gen")
        assert obj.task_type == "code-gen"
"""

from typing import Any, Callable

import pytest
from adapter_registry.models import AdapterRecord
from evolution_svc.models import EvolutionJob
from shared.rune_models import AdapterRef, CodingSession, EvolMetrics
from training_svc.models import TrainingJob


@pytest.fixture
def make_adapter_record() -> Callable[..., AdapterRecord]:
    """Create AdapterRecord instances with defaults."""

    def _factory(**kwargs: Any) -> AdapterRecord:
        defaults: dict[str, Any] = {
            "id": "test-adapter-001",
            "version": 1,
            "task_type": "bug-fix",
            "base_model_id": "Qwen/Qwen2.5-Coder-7B",
            "rank": 16,
            "created_at": "2026-01-01T00:00:00Z",
            "file_path": "/adapters/test-adapter-001.safetensors",
            "file_hash": "abc123def456",
            "file_size_bytes": 1024,
            "pass_rate": None,
            "fitness_score": None,
            "source": "distillation",
            "session_id": "test-session-001",
            "is_archived": False,
        }
        return AdapterRecord(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_adapter_ref() -> Callable[..., AdapterRef]:
    """Create AdapterRef instances with defaults."""

    def _factory(**kwargs: Any) -> AdapterRef:
        defaults: dict[str, Any] = {
            "adapter_id": "test-adapter-001",
            "task_type": "bug-fix",
            "fitness_score": None,
        }
        return AdapterRef(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_coding_session() -> Callable[..., CodingSession]:
    """Create CodingSession instances with defaults."""

    def _factory(**kwargs: Any) -> CodingSession:
        defaults: dict[str, Any] = {
            "session_id": "test-session-001",
            "task_description": "Fix the off-by-one error in list slicing",
            "task_type": "bug-fix",
            "adapter_refs": [],
            "attempt_count": 0,
            "outcome": None,
        }
        return CodingSession(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_training_job() -> Callable[..., TrainingJob]:
    """Create TrainingJob instances with defaults."""

    def _factory(**kwargs: Any) -> TrainingJob:
        defaults: dict[str, Any] = {
            "id": "test-job-001",
            "status": "pending",
            "task_type": "bug-fix",
            "created_at": "2026-01-01T00:00:00Z",
            "adapter_id": None,
        }
        return TrainingJob(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_evolution_job() -> Callable[..., EvolutionJob]:
    """Create EvolutionJob instances with defaults."""

    def _factory(**kwargs: Any) -> EvolutionJob:
        defaults: dict[str, Any] = {
            "id": "test-evol-job-001",
            "status": "pending",
            "task_type": "bug-fix",
            "created_at": "2026-01-01T00:00:00Z",
            "adapter_id": None,
        }
        return EvolutionJob(**{**defaults, **kwargs})

    return _factory


@pytest.fixture
def make_evol_metrics() -> Callable[..., EvolMetrics]:
    """Create EvolMetrics instances with defaults."""

    def _factory(**kwargs: Any) -> EvolMetrics:
        defaults: dict[str, Any] = {
            "adapter_id": "test-adapter-001",
            "pass_rate": 0.75,
            "fitness_score": 0.8,
            "generalization_delta": None,
        }
        return EvolMetrics(**{**defaults, **kwargs})

    return _factory
