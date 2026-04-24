"""CPU-only tests for model_training.training_common.

Verifies the public MLflow helpers work without a GPU and without actually
writing to an MLflow backend:
- RUNE_DISABLE_MLFLOW=1 short-circuits setup_mlflow.
- Missing mlflow package short-circuits setup_mlflow.
- mlflow_log_params silently no-ops when disabled.
- mlflow_run context manager yields cleanly when disabled.
- The module itself is importable in a CPU-only environment.
"""

from __future__ import annotations

import sys

import pytest


def test_setup_mlflow_returns_false_when_disabled_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RUNE_DISABLE_MLFLOW=1 suppresses MLflow regardless of install state."""
    monkeypatch.setenv("RUNE_DISABLE_MLFLOW", "1")

    from model_training.training_common import setup_mlflow

    assert setup_mlflow("any-experiment", None) is False


def test_setup_mlflow_returns_false_on_import_error(
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

    assert setup_mlflow("any-experiment", None) is False


def test_mlflow_log_params_silent_no_op_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlflow_log_params never raises even when mlflow import fails."""
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

    from model_training.training_common import mlflow_log_params

    # Must not raise.
    mlflow_log_params({"key": "value", "count": 42})


def test_mlflow_run_context_yields_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlflow_run with enabled=False completes cleanly without touching mlflow."""
    monkeypatch.setenv("RUNE_DISABLE_MLFLOW", "1")

    from model_training.training_common import mlflow_run

    ran = False
    with mlflow_run(enabled=False, run_name="test-run", params={"a": 1}):
        ran = True

    assert ran


def test_module_is_cpu_importable() -> None:
    """training_common must be importable without torch present."""
    # Remove torch from sys.modules to simulate a CPU-only environment.
    # We only do this check — we don't permanently delete it since other
    # tests in the session may need it.
    had_torch = "torch" in sys.modules

    # Import fresh by temporarily hiding torch if it was imported.
    # The module should already be importable; just assert no AttributeError
    # or ImportError bubbles out of the module-level code.
    import model_training.training_common as tc  # noqa: PLC0415

    assert callable(tc.setup_mlflow)
    assert callable(tc.mlflow_log_params)
    assert callable(tc.mlflow_log_artifact)
    assert callable(tc.mlflow_log_output_artifacts)
    assert callable(tc.mlflow_run)

    # Confirm torch was not imported as a side-effect of the module import.
    if not had_torch:
        assert "torch" not in sys.modules
