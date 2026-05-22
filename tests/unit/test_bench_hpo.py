"""Unit tests for bench HPO module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune.bench.hpo import run_hpo
from rune.bench.runner import BenchResult, BenchTask
from rune.config import PipelineConfig


def _make_task(task_id: str = "t1") -> BenchTask:
    return BenchTask(task_id=task_id, description="write add", test_code="assert True")


def _make_bench_result(pass_at_1: float = 0.8) -> BenchResult:
    return BenchResult(pass_at_1=pass_at_1, total_tasks=5, passed_tasks=4)


def _make_trial(
    adapter_scaling: float = 0.05,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    max_phase_iterations: int = 5,
) -> MagicMock:
    trial = MagicMock()
    trial.suggest_float.side_effect = lambda name, *a, **kw: {
        "adapter_scaling": adapter_scaling,
        "temperature": temperature,
    }[name]
    trial.suggest_int.side_effect = lambda name, *a, **kw: {
        "max_tokens": max_tokens,
        "max_phase_iterations": max_phase_iterations,
    }[name]
    return trial


class TestRunHpoStudyCreation:
    def test_study_created_with_maximize_direction(self) -> None:
        mock_study = MagicMock()
        mock_study.best_params = {"temperature": 0.4}
        mock_study.best_value = 0.8

        with (
            patch("optuna.create_study", return_value=mock_study) as mock_create,
            patch("optuna_integration.MLflowCallback"),
            patch("optuna.logging.set_verbosity"),
            patch(
                "rune.bench.runner.run_benchmark",
                new_callable=AsyncMock,
                return_value=_make_bench_result(),
            ),
            patch("asyncio.to_thread", new_callable=AsyncMock),
        ):
            asyncio.run(run_hpo([_make_task()], MagicMock(), PipelineConfig(), MagicMock(), n_trials=3))

        mock_create.assert_called_once_with(
            direction="maximize", study_name="rune-bench-hpo"
        )


class TestRunHpoMlflowCallback:
    def test_mlflow_callback_attached_with_nested(self) -> None:
        mock_study = MagicMock()
        mock_study.best_params = {}
        mock_study.best_value = 0.0

        mock_cb = MagicMock()

        with (
            patch("optuna.create_study", return_value=mock_study),
            patch("optuna_integration.MLflowCallback", return_value=mock_cb) as mock_cls,
            patch("optuna.logging.set_verbosity"),
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            asyncio.run(run_hpo([], MagicMock(), PipelineConfig(), MagicMock(), n_trials=1))

        mock_cls.assert_called_once_with(mlflow_kwargs={"nested": True})
        # Confirm the callback was passed as keyword arg to study.optimize via to_thread
        call_kwargs = mock_thread.call_args.kwargs
        callbacks_arg = call_kwargs.get("callbacks", [])
        assert mock_cb in callbacks_arg


class TestRunHpoObjectiveProducesFloat:
    def test_objective_returns_pass_at_1_float(self) -> None:
        captured_objective: list = []

        async def fake_to_thread(fn: object, *args: object, **kwargs: object) -> None:
            # to_thread(study.optimize, objective, n_trials, callbacks=[...])
            # fn=study.optimize; args[0]=objective, args[1]=n_trials
            captured_objective.append(args[0])

        mock_study = MagicMock()
        mock_study.best_params = {"temperature": 0.5}
        mock_study.best_value = 0.6

        bench_result = _make_bench_result(pass_at_1=0.6)

        with (
            patch("optuna.create_study", return_value=mock_study),
            patch("optuna_integration.MLflowCallback"),
            patch("optuna.logging.set_verbosity"),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
            patch(
                "rune.bench.runner.run_benchmark",
                new_callable=AsyncMock,
                return_value=bench_result,
            ),
        ):
            asyncio.run(
                run_hpo([_make_task()], MagicMock(), PipelineConfig(), MagicMock(), n_trials=1)
            )

        assert len(captured_objective) == 1
        trial = _make_trial()
        value = captured_objective[0](trial)
        assert isinstance(value, float)
        assert value == pytest.approx(0.6)


class TestRunHpoBestParamsReturned:
    def test_returns_best_params_and_value(self) -> None:
        expected_params = {"temperature": 0.3, "max_tokens": 2048}
        expected_value = 0.9

        mock_study = MagicMock()
        mock_study.best_params = expected_params
        mock_study.best_value = expected_value

        with (
            patch("optuna.create_study", return_value=mock_study),
            patch("optuna_integration.MLflowCallback"),
            patch("optuna.logging.set_verbosity"),
            patch("asyncio.to_thread", new_callable=AsyncMock),
        ):
            result = asyncio.run(
                run_hpo([], MagicMock(), PipelineConfig(), MagicMock(), n_trials=2)
            )

        assert result["best_params"] == expected_params
        assert result["best_value"] == pytest.approx(expected_value)


class TestRunHpoNTrialsPropagated:
    def test_n_trials_passed_to_optimize(self) -> None:
        mock_study = MagicMock()
        mock_study.best_params = {}
        mock_study.best_value = 0.0

        captured: list = []

        async def fake_to_thread(fn: object, *args: object, **kwargs: object) -> None:
            # to_thread(study.optimize, objective, n_trials, callbacks=[...])
            # fn=study.optimize; args[0]=objective, args[1]=n_trials
            captured.append(args)

        with (
            patch("optuna.create_study", return_value=mock_study),
            patch("optuna_integration.MLflowCallback"),
            patch("optuna.logging.set_verbosity"),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
        ):
            asyncio.run(
                run_hpo([], MagicMock(), PipelineConfig(), MagicMock(), n_trials=7)
            )

        # args[0]=objective_fn, args[1]=n_trials
        assert captured[0][1] == 7
