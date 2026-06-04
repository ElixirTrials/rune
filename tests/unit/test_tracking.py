"""Tests for rune.tracking dataset logging."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from rune.tracking import log_dataset


@pytest.fixture
def sqlite_mlflow(tmp_path: Path):
    """Real MLflow tracking against a throwaway sqlite backend (3.13 file-store
    is maintenance-mode; sqlite is the supported local backend)."""
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-dataset-logging")
    yield
    mlflow.set_tracking_uri(prev)


def test_log_dataset_records_input_with_name_context_and_source(
    sqlite_mlflow: None,
) -> None:
    uri = "s3://bucket/datasets/external_codereview.val.clean.jsonl"

    with mlflow.start_run() as run:
        digest = log_dataset(uri, name="external_codereview.val", context="validation")
        run_id = run.info.run_id

    logged = mlflow.get_run(run_id).inputs.dataset_inputs
    assert len(logged) == 1
    di = logged[0]
    assert di.dataset.name == "external_codereview.val"
    assert di.dataset.source_type == "s3"
    assert uri in di.dataset.source
    assert di.dataset.digest == digest  # returns MLflow's computed digest
    assert ("mlflow.data.context", "validation") in {(t.key, t.value) for t in di.tags}
