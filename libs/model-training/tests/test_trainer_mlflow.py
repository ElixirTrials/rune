"""CPU tests for MLflow integration in trainer.py.

Verifies the gating helpers work without a GPU and without actually writing
to an MLflow backend:
- RUNE_DISABLE_MLFLOW=1 short-circuits setup.
- Missing ``mlflow`` package short-circuits setup.
- ``_mlflow_log_params`` / ``_mlflow_log_artifact`` silently no-op when
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

    from model_training.trainer import _setup_mlflow_trainer

    assert _setup_mlflow_trainer("any-experiment", tracking_uri=None) is False


def test_setup_returns_false_when_mlflow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent mlflow module returns False without raising."""
    monkeypatch.delenv("RUNE_DISABLE_MLFLOW", raising=False)

    # Force ImportError on `import mlflow` inside the helper.
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

    # Drop any cached mlflow module so our fake_import takes effect.
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    from model_training.trainer import _setup_mlflow_trainer

    assert _setup_mlflow_trainer("any-experiment", tracking_uri=None) is False


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

    from model_training.trainer import _mlflow_log_artifact, _mlflow_log_params

    # Must not raise.
    _mlflow_log_params({"k": 1})
    _mlflow_log_artifact("/nonexistent/path")


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
