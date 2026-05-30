"""Unit tests for orchestrator pipeline stages and HPO."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rune.training.d2l_train import D2LTrainConfig
from rune.training.orchestrator import (
    _run_hypernetwork_distillation,
    _run_success_gate,
    run_training_pipeline,
)


@pytest.fixture()
def base_config() -> D2LTrainConfig:
    return D2LTrainConfig()


def test_stage2_dispatches_to_hypernet_distill() -> None:
    with patch("rune.training.hypernet_distill.run_hypernet_distillation") as m:
        _run_hypernetwork_distillation(config=object())
        m.assert_called_once()


class TestRunSuccessGate:
    def test_pass(self) -> None:
        baseline = {"humaneval": 0.50, "mbpp": 0.40, "apps": 0.30, "ds1000": 0.20}
        new = {
            "humaneval": 0.55,
            "mbpp": 0.45,
            "apps": 0.35,
            "ds1000": 0.25,
            "swebench": 0.30,
        }
        assert _run_success_gate(baseline, new) == 0

    def test_fail(self) -> None:
        assert _run_success_gate({}, {}) == 1


class TestRunTrainingPipeline:
    @pytest.mark.asyncio
    async def test_single_run_calls_distill(
        self, base_config: D2LTrainConfig, tmp_path: Path
    ) -> None:
        with patch(
            "rune.training.orchestrator._run_hypernetwork_distillation"
        ) as mock_distill:
            result = await run_training_pipeline(base_config, tmp_path)

        mock_distill.assert_called_once_with(base_config)
        assert result == 0

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
        mock_optuna_integration.MLflowCallback = mock_mlflow_cb

        with (
            patch.dict(
                sys.modules,
                {
                    "optuna": mock_optuna,
                    "optuna_integration": mock_optuna_integration,
                },
            ),
            patch("rune.training.orchestrator._run_hypernetwork_distillation"),
        ):
            result = await run_training_pipeline(
                base_config, tmp_path, hpo=True, n_trials=3
            )

        mock_optuna.create_study.assert_called_once_with(direction="minimize")
        mock_study.optimize.assert_called_once()
        assert result == 0
