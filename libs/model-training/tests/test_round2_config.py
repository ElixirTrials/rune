"""Pydantic validator tests for Round2TrainConfig."""

from __future__ import annotations

import pytest
from model_training.round2_config import (
    DEFAULT_MAX_LOADED_ORACLES,
    DEFAULT_MIN_ORACLE_COVERAGE,
    DEFAULT_ROUND2_CHECKPOINT_DIR,
    DEFAULT_ROUND2_EXPERIMENT_NAME,
    Round2TrainConfig,
)


def _minimal_kwargs(**overrides: object) -> dict[str, object]:
    """Build a minimal valid kwargs dict for Round2TrainConfig."""
    base: dict[str, object] = {
        "sakana_checkpoint_path": "/tmp/fake.bin",
        "oracle_registry_url": "sqlite:///fake.db",
    }
    base.update(overrides)
    return base


def test_round2_config_defaults_are_sane() -> None:
    """Defaults match the constants declared at module scope."""
    cfg = Round2TrainConfig(**_minimal_kwargs())
    assert cfg.max_loaded_oracles == DEFAULT_MAX_LOADED_ORACLES
    assert cfg.min_oracle_coverage == DEFAULT_MIN_ORACLE_COVERAGE
    assert cfg.oracle_fallback == "skip"
    assert cfg.checkpoint_dir == DEFAULT_ROUND2_CHECKPOINT_DIR
    assert cfg.experiment_name == DEFAULT_ROUND2_EXPERIMENT_NAME
    assert cfg.sakana_checkpoint_path == "/tmp/fake.bin"
    assert cfg.oracle_registry_url == "sqlite:///fake.db"


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_round2_config_rejects_non_positive_max_loaded(bad_value: int) -> None:
    """max_loaded_oracles must be >= 1."""
    with pytest.raises(ValueError, match="max_loaded_oracles must be >= 1"):
        Round2TrainConfig(**_minimal_kwargs(max_loaded_oracles=bad_value))


def test_round2_config_rejects_coverage_out_of_range() -> None:
    """min_oracle_coverage must be in [0.0, 1.0]."""
    with pytest.raises(ValueError, match="min_oracle_coverage must be in"):
        Round2TrainConfig(**_minimal_kwargs(min_oracle_coverage=1.5))
    with pytest.raises(ValueError, match="min_oracle_coverage must be in"):
        Round2TrainConfig(**_minimal_kwargs(min_oracle_coverage=-0.1))


def test_round2_config_rejects_unknown_fallback() -> None:
    """oracle_fallback must be 'base_model' or 'skip'."""
    with pytest.raises(ValueError, match="'base_model'|'skip'"):
        Round2TrainConfig(**_minimal_kwargs(oracle_fallback="nope"))


def test_round2_config_inherits_d2l_fields() -> None:
    """Inherits lr, alpha, temperature, etc. from D2LTrainConfig."""
    cfg = Round2TrainConfig(**_minimal_kwargs(lr=1e-4, alpha=0.7))
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.alpha == pytest.approx(0.7)
    # default from parent
    assert cfg.temperature == pytest.approx(2.0)
