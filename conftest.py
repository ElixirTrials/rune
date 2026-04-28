"""Root conftest.py — shared factory fixtures for all Rune test suites.

This file lives at the repo root so pytest auto-discovers it for every test
under services/ and libs/. Each fixture returns a factory function that accepts
**kwargs for field overrides; unspecified fields use deterministic defaults.

Usage (no import needed in test files):

    def test_something(make_adapter_record):
        obj = make_adapter_record(task_type="code-gen")
        assert obj.task_type == "code-gen"
"""

import importlib.util
import logging
from typing import Any, Callable, TypeVar

# Patch torch's _dispatch_library to be idempotent — prevents the
# _TritonLibrary double-registration crash in pytest-xdist workers.
if importlib.util.find_spec("torch") is not None:
    try:
        import torch._C  # noqa: F401

        _orig_dispatch_library = torch._C._dispatch_library

        def _safe_dispatch_library(
            kind: str, ns: str, dispatch_key: str, filename: str, lineno: int
        ) -> object:
            try:
                return _orig_dispatch_library(kind, ns, dispatch_key, filename, lineno)
            except RuntimeError as e:
                if "Only a single TORCH_LIBRARY" in str(e):
                    return None
                raise

        torch._C._dispatch_library = _safe_dispatch_library  # type: ignore[assignment]
    except (ImportError, AttributeError):
        logging.getLogger(__name__).debug(
            "torch dispatch library patch skipped", exc_info=True
        )

import pytest
from adapter_registry.models import AdapterRecord
from evolution_svc.models import EvolutionJob
from shared.rune_models import (
    AdapterRef,
    CodingSession,
    EvolMetrics,
    SwarmCheckpoint,
    SwarmConfig,
)
from training_svc.models import TrainingJob

# ---------------------------------------------------------------------------
# xdist grouping: torch tests must run on the same worker to avoid the
# triton library double-registration crash in pytest-xdist.
# ---------------------------------------------------------------------------
_TORCH_TEST_FILENAMES = {
    "test_hypernetwork",
    "test_merging",
    "test_sakana",
    "test_d2l_config",
    "test_d2l_train",
    "test_d2l_weight_transfer",
    "test_d2l_probe",
    "test_d2l_prep",
    "test_trainer",
    "test_transformers_provider",
    "test_training",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Group torch-dependent test files onto one xdist worker.

    Matches by stem (without .py) from the node ID to avoid path
    resolution differences across pytest import modes.
    """
    for item in items:
        # nodeid looks like "libs/model-training/tests/test_hypernetwork.py::test_foo"
        parts = item.nodeid.split("::")
        if parts:
            file_stem = parts[0].rsplit("/", 1)[-1].removesuffix(".py")
            if file_stem in _TORCH_TEST_FILENAMES:
                item.add_marker(pytest.mark.xdist_group("torch"))


T = TypeVar("T")


def _make_factory(cls: type[T], defaults: dict[str, Any]) -> Callable[..., T]:
    """Return a factory callable that creates *cls* instances with merged defaults."""

    def _factory(**kwargs: Any) -> T:
        return cls(**{**defaults, **kwargs})  # type: ignore[call-arg]

    return _factory


@pytest.fixture
def make_adapter_record() -> Callable[..., AdapterRecord]:
    """Create AdapterRecord instances with defaults."""
    return _make_factory(
        AdapterRecord,
        {
            "id": "test-adapter-001",
            "version": 1,
            "task_type": "bug-fix",
            "base_model_id": "Qwen/Qwen3.5-9B",
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
            "parent_ids": None,
            "generation": 0,
            "training_task_hash": None,
            "agent_id": None,
        },
    )


@pytest.fixture
def make_adapter_ref() -> Callable[..., AdapterRef]:
    """Create AdapterRef instances with defaults."""
    return _make_factory(
        AdapterRef,
        {
            "adapter_id": "test-adapter-001",
            "task_type": "bug-fix",
            "fitness_score": None,
        },
    )


@pytest.fixture
def make_coding_session() -> Callable[..., CodingSession]:
    """Create CodingSession instances with defaults."""
    return _make_factory(
        CodingSession,
        {
            "session_id": "test-session-001",
            "task_description": "Fix the off-by-one error in list slicing",
            "task_type": "bug-fix",
            "adapter_refs": [],
            "attempt_count": 0,
            "outcome": None,
        },
    )


@pytest.fixture
def make_training_job() -> Callable[..., TrainingJob]:
    """Create TrainingJob instances with defaults."""
    return _make_factory(
        TrainingJob,
        {
            "id": "test-job-001",
            "status": "pending",
            "task_type": "bug-fix",
            "created_at": "2026-01-01T00:00:00Z",
            "adapter_id": None,
        },
    )


@pytest.fixture
def make_evolution_job() -> Callable[..., EvolutionJob]:
    """Create EvolutionJob instances with defaults."""
    return _make_factory(
        EvolutionJob,
        {
            "id": "test-evol-job-001",
            "status": "pending",
            "task_type": "bug-fix",
            "created_at": "2026-01-01T00:00:00Z",
            "adapter_id": None,
        },
    )


@pytest.fixture
def make_evol_metrics() -> Callable[..., EvolMetrics]:
    """Create EvolMetrics instances with defaults."""
    return _make_factory(
        EvolMetrics,
        {
            "adapter_id": "test-adapter-001",
            "pass_rate": 0.75,
            "fitness_score": 0.8,
            "generalization_delta": None,
        },
    )


@pytest.fixture
def make_swarm_config() -> Callable[..., SwarmConfig]:
    """Create SwarmConfig instances with defaults."""
    return _make_factory(
        SwarmConfig,
        {
            "db_url": "sqlite:///:memory:",
            "task_source": "tasks.json",
            "population_size": 4,
            "max_generations": 2,
            "evolution_interval": 60,
            "sandbox_backend": "subprocess",
            "base_model_id": "Qwen/Qwen3.5-9B",
        },
    )


@pytest.fixture
def make_swarm_checkpoint() -> Callable[..., SwarmCheckpoint]:
    """Create SwarmCheckpoint instances with defaults."""
    return _make_factory(
        SwarmCheckpoint,
        {
            "run_id": "test-run-001",
            "task_hash": "hash-001",
            "agent_id": "agent-001",
            "status": "pending",
            "outcome": None,
            "started_at": None,
            "completed_at": None,
        },
    )
