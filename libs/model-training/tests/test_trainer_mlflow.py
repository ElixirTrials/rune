"""CPU tests for MLflow integration in training_common + trainer kwargs.

Verifies the gating helpers work without a GPU and without actually writing
to an MLflow backend:
- RUNE_DISABLE_MLFLOW=1 short-circuits setup.
- Missing ``mlflow`` package short-circuits setup.
- ``mlflow_log_params`` / ``mlflow_log_artifact`` silently no-op when
  tracking is disabled so training never breaks.
- ``train_qlora`` accepts ``mlflow_experiment`` and ``mlflow_tracking_uri``
  kwargs without touching the GPU.
"""

from __future__ import annotations

import sys

import pytest


def test_setup_returns_false_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RUNE_DISABLE_MLFLOW=1 suppresses MLflow regardless of install state."""
    monkeypatch.setenv("RUNE_DISABLE_MLFLOW", "1")

    from model_training.training_common import setup_mlflow

    assert setup_mlflow("any-experiment", tracking_uri=None) is False


def test_setup_returns_false_when_mlflow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent mlflow module returns False without raising."""
    monkeypatch.delenv("RUNE_DISABLE_MLFLOW", raising=False)

    real_import = (
        __builtins__["__import__"]  # type: ignore[index]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__  # type: ignore[attr-defined]
    )

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlflow":
            raise ImportError("mocked: mlflow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    from model_training.training_common import setup_mlflow

    assert setup_mlflow("any-experiment", tracking_uri=None) is False


def test_log_helpers_silent_noop_when_mlflow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The log helpers never raise, even if mlflow import fails mid-call."""
    real_import = (
        __builtins__["__import__"]  # type: ignore[index]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__  # type: ignore[attr-defined]
    )

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlflow":
            raise ImportError("mocked: mlflow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    from model_training.training_common import mlflow_log_artifact, mlflow_log_params

    mlflow_log_params({"k": 1})
    mlflow_log_artifact("/nonexistent/path")


def test_train_qlora_accepts_mlflow_kwargs() -> None:
    """train_qlora exposes mlflow_experiment and mlflow_tracking_uri kwargs."""
    import inspect

    from model_training.trainer import train_qlora

    sig = inspect.signature(train_qlora)
    assert "mlflow_experiment" in sig.parameters
    assert "mlflow_tracking_uri" in sig.parameters
    # Defaults preserve backward-compatible behavior: rune-qlora experiment,
    # URI resolved from env/fallback at runtime.
    assert sig.parameters["mlflow_experiment"].default == "rune-qlora"
    assert sig.parameters["mlflow_tracking_uri"].default is None


def test_train_and_register_accepts_mlflow_kwargs() -> None:
    """train_and_register forwards mlflow kwargs."""
    import inspect

    from model_training.trainer import train_and_register

    sig = inspect.signature(train_and_register)
    assert "mlflow_experiment" in sig.parameters
    assert "mlflow_tracking_uri" in sig.parameters


# ---------------------------------------------------------------------------
# assistant_masking_strategy in MLflow run_params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "diff_aware_loss,expected_strategy",
    [(False, "assistant_masks"), (True, "diff_weighted")],
)
def test_run_params_assistant_masking_strategy(
    diff_aware_loss: bool, expected_strategy: str
) -> None:
    """assistant_masking_strategy reflects which masking path is in effect."""
    from model_training.trainer import _build_run_params

    params = _build_run_params(
        model_id="test-model",
        warm_start=None,
        resolved_rank=64,
        resolved_alpha=128,
        resolved_epochs=3,
        learning_rate=2e-4,
        resolved_grad_accum=16,
        resolved_lr_sched="constant",
        attn_impl=None,
        dataset_size=10,
        diff_aware_loss=diff_aware_loss,
        task_type="code-gen",
        adapter_id="test-adapter",
        session_id=None,
        dataset_path=None,
        encoding_mode="multi_turn",
        diff_changed_weight=1.0,
        diff_unchanged_weight=0.3,
        override_lora_alpha=None,
        override_lora_dropout=None,
        neftune_noise_alpha=None,
    )
    assert params["assistant_masking_strategy"] == expected_strategy
    assert params["diff_aware_loss"] is diff_aware_loss
    # warmup_ratio and assistant_only_loss are no longer logged here — TRL
    # owns those fields via SFTConfig + MLflowCallback. Logging them with our
    # values triggered async_logging_queue overwrite errors.
    assert "warmup_ratio" not in params
    assert "assistant_only_loss" not in params
    # None values are skipped defensively.
    assert "override_lora_alpha" not in params
    assert "override_lora_dropout" not in params
    assert "requested_neftune_noise_alpha" not in params


def test_run_params_includes_requested_neftune_when_set() -> None:
    """When neftune is set, it surfaces under requested_* to avoid TRL collision."""
    from model_training.trainer import _build_run_params

    params = _build_run_params(
        model_id="test-model",
        warm_start=None,
        resolved_rank=64,
        resolved_alpha=128,
        resolved_epochs=3,
        learning_rate=2e-4,
        resolved_grad_accum=16,
        resolved_lr_sched="constant",
        attn_impl=None,
        dataset_size=10,
        diff_aware_loss=False,
        task_type="code-gen",
        adapter_id="test-adapter",
        session_id=None,
        dataset_path=None,
        encoding_mode="multi_turn",
        diff_changed_weight=1.0,
        diff_unchanged_weight=0.3,
        override_lora_alpha=None,
        override_lora_dropout=None,
        neftune_noise_alpha=5.0,
    )
    assert params["requested_neftune_noise_alpha"] == 5.0
    assert "neftune_noise_alpha" not in params
