"""Lightweight MLflow setup. One configure call + one context manager."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mlflow
import mlflow.langchain


def configure_mlflow(experiment: str) -> None:
    """One-time MLflow setup: dotenv, tracking URI, experiment, autolog."""
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    mlflow.langchain.autolog()


@contextmanager
def tracked_run(
    name: str,
    params: dict[str, Any] | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager: start_run + log_params. Yields the ActiveRun."""
    with mlflow.start_run(run_name=name) as run:
        if params:
            mlflow.log_params(params)
        yield run
