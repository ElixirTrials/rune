"""Pydantic config for round-2 hypernetwork distillation training.

Inherits every round-1 field from :class:`model_training.d2l_train.D2LTrainConfig`
and adds oracle-adapter routing fields. Defaults keep behaviour as close as
possible to round-1 so operators can diff-review a round-1 vs round-2 run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from model_training.d2l_train import D2LTrainConfig

DEFAULT_MAX_LOADED_ORACLES: int = 4
DEFAULT_MIN_ORACLE_COVERAGE: float = 0.8
DEFAULT_ROUND2_CHECKPOINT_DIR: str = "./checkpoints/round2"
DEFAULT_ROUND2_EXPERIMENT_NAME: str = "d2l-qwen3-round2"


class Round2TrainConfig(D2LTrainConfig):
    """Configuration for round-2 (oracle-teacher) hypernetwork training.

    Attributes:
        oracle_registry_url: SQLAlchemy URL for the AdapterRegistry SQLite DB
            that holds the 25 per-bin oracle adapter records.
        max_loaded_oracles: LRU cache cap for simultaneously cached oracle
            LoRA dicts (functional-LoRA format: ``{module: {A, B}}``). Each
            entry is lightweight (tensors only, no model wrappers); this bound
            mostly limits the rate of disk reads.
        min_oracle_coverage: Minimum fraction of training records that must
            route to a registered oracle. When the startup audit reports less
            than this, training aborts. Effectively mandatory because the
            default fallback (``"skip"``) means below-coverage runs make no
            training progress.
        oracle_fallback: What to do when a record's bin has no registered
            oracle. ``"skip"`` (default) drops the record from the epoch —
            preserves the premise that the hypernet should only learn from
            oracle signal. ``"base_model"`` uses the bare base model (round-1
            behaviour); kept available for ablations.
        checkpoint_dir: Overrides parent default so round-2 does not clobber
            round-1 checkpoints.
        experiment_name: Overrides parent default so MLflow separates the runs.
    """

    oracle_registry_url: str
    max_loaded_oracles: int = Field(default=DEFAULT_MAX_LOADED_ORACLES)
    min_oracle_coverage: float = Field(default=DEFAULT_MIN_ORACLE_COVERAGE)
    oracle_fallback: Literal["base_model", "skip"] = Field(default="skip")
    checkpoint_dir: str = Field(default=DEFAULT_ROUND2_CHECKPOINT_DIR)
    experiment_name: str = Field(default=DEFAULT_ROUND2_EXPERIMENT_NAME)

    @field_validator("max_loaded_oracles")
    @classmethod
    def _validate_max_loaded(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_loaded_oracles must be >= 1, got {v}")
        return v

    @field_validator("min_oracle_coverage")
    @classmethod
    def _validate_coverage(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"min_oracle_coverage must be in [0.0, 1.0], got {v}")
        return v
