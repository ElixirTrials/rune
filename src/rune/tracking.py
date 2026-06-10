"""Lightweight MLflow setup. One configure call + one context manager."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.langchain
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.meta_dataset import MetaDataset


def configure_mlflow(experiment: str) -> None:
    """One-time MLflow setup: dotenv, tracking URI, experiment, autolog."""
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    mlflow.langchain.autolog()


def log_dataset(uri: str | Path, *, name: str, context: str) -> str:
    """Log a dataset (file/dir/S3 URI) as an MLflow input so the run is reproducible.

    A metadata-only ``MetaDataset`` records the source + MLflow's content digest by
    reference — it does not read the data, so it is safe for large corpora. ``uri`` is
    the durable canonical location (prefer ``s3://…``); ``context`` is one of
    ``training`` / ``validation`` / ``test``. Returns the digest.
    """
    # MetaDataset is concrete at runtime; MLflow's ABC stubs trip mypy's abstract check.
    dataset = MetaDataset(source=resolve_dataset_source(str(uri)), name=name)  # type: ignore[abstract]
    mlflow.log_input(dataset, context=context)
    return dataset.digest


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
