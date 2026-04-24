"""Shared MLflow helpers for model-training modules.

All MLflow imports are deferred inside function bodies to ensure CPU-only
importability (INFRA-05).  Module-level imports: stdlib only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str, tracking_uri: str | None) -> bool:
    """Configure MLflow for training runs.

    Returns True when MLflow is usable and configured; False when tracking
    should be skipped. Skipping happens when RUNE_DISABLE_MLFLOW=1 is set
    in the environment, or when mlflow itself is not importable.

    Tracking URI precedence: explicit ``tracking_uri`` arg, then the
    ``MLFLOW_TRACKING_URI`` env var, then ``./mlruns`` as a local-dev fallback.

    Args:
        experiment_name: MLflow experiment name to activate.
        tracking_uri: Optional explicit tracking URI override.

    Returns:
        True if MLflow is active and configured, False otherwise.
    """
    if os.environ.get("RUNE_DISABLE_MLFLOW") == "1":
        return False
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        return False
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow enabled: tracking_uri=%s experiment=%s", uri, experiment_name)
    return True


def mlflow_log_params(params: dict[str, Any]) -> None:
    """Log a dict of params to the active MLflow run. Silent no-op on failure.

    All exceptions are caught and logged at DEBUG level so that MLflow
    unavailability never interrupts training.

    Args:
        params: Mapping of parameter names to values forwarded verbatim to
            ``mlflow.log_params``.

    Returns:
        None.

    Raises:
        None: All exceptions from ``mlflow.log_params`` are caught and
            logged at DEBUG level; MLflow unavailability never interrupts
            training.
    """
    try:
        import mlflow  # noqa: PLC0415

        mlflow.log_params(params)
    except Exception:  # noqa: BLE001 — logging must never break training
        logger.debug("mlflow.log_params skipped", exc_info=True)


def mlflow_log_artifact(path: str) -> None:
    """Log a file artifact to the active MLflow run. Silent no-op on failure."""
    try:
        import mlflow  # noqa: PLC0415

        mlflow.log_artifact(path)
    except Exception:  # noqa: BLE001
        logger.debug("mlflow.log_artifact skipped for %s", path, exc_info=True)


def mlflow_log_output_artifacts(output_dir: str) -> None:
    """Log the saved adapter's safetensors + config.json to MLflow, if present."""
    adapter_safetensors = Path(output_dir) / "adapter_model.safetensors"
    adapter_config = Path(output_dir) / "adapter_config.json"
    if adapter_safetensors.exists():
        mlflow_log_artifact(str(adapter_safetensors))
    if adapter_config.exists():
        mlflow_log_artifact(str(adapter_config))


@contextmanager
def mlflow_run(
    *, enabled: bool, run_name: str, params: dict[str, Any]
) -> Iterator[None]:
    """Context manager that starts an MLflow run when enabled, else no-ops.

    When enabled, logs ``params`` at entry and ensures ``mlflow.end_run()`` on
    exit even if training raises. When disabled, the body runs unchanged.

    If an MLflow run is already active in the current thread (e.g. the HPO
    harness opened one to attach study-level tags), attach params to it
    instead of starting a new run — MLflow permits only one top-level run
    per thread, and forcing a nested run here would hide trainer metrics
    inside an extra layer the caller didn't ask for.

    Args:
        enabled: Whether MLflow tracking is active for this run.
        run_name: Display name for the MLflow run.
        params: Parameter dict to log at run start.
    """
    if not enabled:
        yield
        return
    import mlflow  # noqa: PLC0415

    # Respect an already-active run: the caller owns it and is responsible
    # for closing it. We just decorate it with params.
    if mlflow.active_run() is not None:
        mlflow_log_params(params)
        yield
        return

    mlflow.start_run(run_name=run_name)
    try:
        mlflow_log_params(params)
        yield
    finally:
        mlflow.end_run()
