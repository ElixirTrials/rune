"""Unit tests for orchestrator pipeline stages and HPO."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rune.training.d2l_train import D2LTrainConfig
from rune.training.oracle_cache import audit_oracle_coverage, lookup_oracle_path
from rune.training.orchestrator import (
    _run_oracle_training,
    _run_success_gate,
    run_training_pipeline,
)


@pytest.fixture()
def base_config() -> D2LTrainConfig:
    return D2LTrainConfig()


class TestRunOracleTraining:
    def test_runs_without_error(self, base_config: D2LTrainConfig, tmp_path: Path) -> None:
        _run_oracle_training(base_config, tmp_path)


class TestRunSuccessGate:
    def test_pass(self) -> None:
        baseline = {"humaneval": 50.0, "mbpp": 40.0, "apps": 30.0, "ds1000": 20.0}
        new = {
            "humaneval": 55.0,
            "mbpp": 45.0,
            "apps": 35.0,
            "ds1000": 25.0,
            "swebench": 30.0,
        }
        assert _run_success_gate(baseline, new) == 0

    def test_fail(self) -> None:
        assert _run_success_gate({}, {}) == 1


class TestRunTrainingPipeline:
    @pytest.mark.asyncio
    async def test_single_run_calls_stages(
        self, base_config: D2LTrainConfig, tmp_path: Path
    ) -> None:
        with (
            patch("rune.training.orchestrator._run_oracle_training") as mock_oracle,
            patch(
                "rune.training.orchestrator._run_hypernetwork_distillation"
            ) as mock_distill,
            patch(
                "rune.training.orchestrator._run_success_gate", return_value=0
            ) as mock_gate,
        ):
            result = await run_training_pipeline(base_config, tmp_path)

        mock_oracle.assert_called_once_with(base_config, tmp_path)
        mock_distill.assert_called_once_with(base_config)
        mock_gate.assert_called_once()
        assert result == 0

    @pytest.mark.asyncio
    async def test_single_run_gate_failure_returns_1(
        self, base_config: D2LTrainConfig, tmp_path: Path
    ) -> None:
        with (
            patch("rune.training.orchestrator._run_oracle_training"),
            patch("rune.training.orchestrator._run_hypernetwork_distillation"),
            patch("rune.training.orchestrator._run_success_gate", return_value=1),
        ):
            result = await run_training_pipeline(base_config, tmp_path)

        assert result == 1

    @pytest.mark.asyncio
    async def test_hpo_creates_study(
        self, base_config: D2LTrainConfig, tmp_path: Path
    ) -> None:
        mock_study = MagicMock()
        mock_study.best_value = 0.0
        mock_study.best_params = {"learning_rate": 1e-5}

        mock_optuna = MagicMock()
        mock_optuna.create_study.return_value = mock_study

        mock_mlflow_cb = MagicMock()
        mock_optuna_integration = MagicMock()
        mock_optuna_integration.mlflow.MLflowCallback = mock_mlflow_cb

        with (
            patch.dict(
                sys.modules,
                {
                    "optuna": mock_optuna,
                    "optuna.integration": mock_optuna_integration,
                    "optuna.integration.mlflow": mock_optuna_integration.mlflow,
                },
            ),
            patch("rune.training.orchestrator._run_oracle_training"),
            patch("rune.training.orchestrator._run_hypernetwork_distillation"),
            patch("rune.training.orchestrator._run_success_gate", return_value=0),
        ):
            result = await run_training_pipeline(
                base_config, tmp_path, hpo=True, n_trials=3
            )

        mock_optuna.create_study.assert_called_once_with(direction="minimize")
        mock_study.optimize.assert_called_once()
        assert result == 0


class TestOracleCacheIntegration:
    """Verify oracle_cache uses the v2 registry interface."""

    def test_lookup_oracle_path_uses_get(self) -> None:
        mock_registry = MagicMock()
        record = MagicMock()
        record.disk_path = "/path/to/adapter"
        mock_registry.get.return_value = record

        result = lookup_oracle_path("decompose_humaneval", mock_registry)

        mock_registry.get.assert_called_once_with("oracle_decompose_humaneval")
        assert result == "/path/to/adapter"

    def test_lookup_oracle_path_returns_none_when_missing(self) -> None:
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        result = lookup_oracle_path("missing_bin", mock_registry)

        assert result is None

    def test_audit_coverage_with_v2_registry(self) -> None:
        mock_registry = MagicMock()
        record = MagicMock()
        record.disk_path = "/path/to/adapter"
        mock_registry.get.return_value = record

        records = [
            {"task_id": "humaneval/001/decompose"},
            {"task_id": "mbpp/002/implement"},
        ]
        coverage, bin_counts = audit_oracle_coverage(records, mock_registry)

        assert coverage == pytest.approx(1.0)
        assert len(bin_counts) == 2
