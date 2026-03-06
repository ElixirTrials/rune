"""Tests for training-svc endpoint routes."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_job_store():
    """Clear JOB_STORE before/after each test to prevent cross-test contamination."""
    from training_svc.jobs import JOB_STORE

    JOB_STORE.clear()
    yield
    JOB_STORE.clear()


def test_train_lora_returns_job_id(test_client):
    """POST /train/lora returns 200 with job_id and status=queued."""
    with patch("training_svc.routers.training._run_training_job") as mock_run:
        mock_run.return_value = None
        response = test_client.post(
            "/train/lora",
            json={
                "session_id": "test-session",
                "task_type": "code-gen",
                "rank": 64,
                "epochs": 3,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_train_lora_requires_session_id(test_client):
    """POST /train/lora without session_id returns 422 validation error."""
    response = test_client.post(
        "/train/lora",
        json={"task_type": "code-gen"},
    )
    assert response.status_code == 422


def test_get_job_status_found(test_client):
    """GET /jobs/{job_id} returns status for an existing job."""
    with patch("training_svc.routers.training._run_training_job") as mock_run:
        mock_run.return_value = None
        post_response = test_client.post(
            "/train/lora",
            json={
                "session_id": "test-session",
                "task_type": "code-gen",
                "rank": 64,
                "epochs": 3,
            },
        )
    assert post_response.status_code == 200
    job_id = post_response.json()["job_id"]

    get_response = test_client.get(f"/jobs/{job_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"


def test_get_job_status_not_found(test_client):
    """GET /jobs/{nonexistent-id} returns 404."""
    response = test_client.get("/jobs/nonexistent-id")
    assert response.status_code == 404


def test_train_hypernetwork_returns_job_id(test_client):
    """POST /train/hypernetwork returns 200 with job_id and status=queued."""
    with patch("training_svc.routers.training._run_hypernetwork_job") as mock_run:
        mock_run.return_value = None
        response = test_client.post(
            "/train/hypernetwork",
            json={"task_type": "gen", "trajectory_ids": ["t-1"]},
        )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_train_hypernetwork_requires_trajectory_ids(test_client):
    """POST /train/hypernetwork without trajectory_ids returns 422 validation error."""
    response = test_client.post(
        "/train/hypernetwork",
        json={"task_type": "gen"},
    )
    assert response.status_code == 422


def test_train_hypernetwork_job_pollable(test_client):
    """After POST /train/hypernetwork, GET /jobs/{job_id} returns the job status."""
    with patch("training_svc.routers.training._run_hypernetwork_job") as mock_run:
        mock_run.return_value = None
        post_response = test_client.post(
            "/train/hypernetwork",
            json={"task_type": "gen", "trajectory_ids": ["t-1"]},
        )
    assert post_response.status_code == 200
    job_id = post_response.json()["job_id"]

    get_response = test_client.get(f"/jobs/{job_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"


def test_hypernetwork_job_registers_adapter(tmp_path, monkeypatch):
    """_run_hypernetwork_job stores AdapterRecord in registry after saving adapter."""
    from adapter_registry.registry import AdapterRegistry
    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel, create_engine
    from training_svc.jobs import JOB_STORE, JobStatus
    from training_svc.routers.training import _run_hypernetwork_job

    # -- Set up test database --
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(bind=test_engine)

    # -- Configure env vars --
    adapter_dir_base = str(tmp_path)
    monkeypatch.setenv("RUNE_ADAPTER_DIR", adapter_dir_base)
    monkeypatch.setenv("RUNE_BASE_MODEL", "test-model")
    dummy_weights = tmp_path / "hypernetwork.pt"
    dummy_weights.write_bytes(b"fake-weights")
    monkeypatch.setenv("RUNE_HYPERNETWORK_WEIGHTS_PATH", str(dummy_weights))

    # -- Pre-populate JOB_STORE --
    job_id = "test-job-hypernetwork"
    adapter_id = "test-adapter-hypernetwork"
    trajectory_id = "test-trajectory-001"
    task_type = "code-gen"

    JOB_STORE[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        adapter_id=adapter_id,
    )

    # -- Inject GPU mocks via sys.modules (INFRA-05 pattern) --
    # torch mock
    torch_mod = ModuleType("torch")
    torch_mod.no_grad = MagicMock(
        return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None)
    )  # type: ignore[attr-defined]
    fake_weights_tensor = MagicMock()
    torch_mod.load = MagicMock(return_value={})  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_mod

    # transformers mock
    transformers_mod = ModuleType("transformers")
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = MagicMock()
    mock_tokenizer.vocab_size = 32000
    mock_auto_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    transformers_mod.AutoTokenizer = mock_auto_tokenizer  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers_mod

    # model_training.hypernetwork mock — save_hypernetwork_adapter must create the file
    def fake_save_hypernetwork_adapter(weights, output_dir, base_model_id):
        import os

        os.makedirs(output_dir, exist_ok=True)
        adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
        with open(adapter_file, "wb") as f:
            f.write(b"fake-adapter-data-for-testing")

    mock_hypernetwork_instance = MagicMock()
    mock_hypernetwork_instance.return_value = fake_weights_tensor
    mock_hypernetwork_class = MagicMock(return_value=mock_hypernetwork_instance)

    hypernetwork_mod = ModuleType("model_training.hypernetwork")
    hypernetwork_mod.DocToLoraHypernetwork = mock_hypernetwork_class  # type: ignore[attr-defined]
    hypernetwork_mod.save_hypernetwork_adapter = fake_save_hypernetwork_adapter  # type: ignore[attr-defined]
    sys.modules["model_training"] = ModuleType("model_training")
    sys.modules["model_training.hypernetwork"] = hypernetwork_mod

    # model_training.trajectory mock
    trajectory_mod = ModuleType("model_training.trajectory")
    trajectory_mod.load_trajectory = MagicMock(return_value={"id": trajectory_id})  # type: ignore[attr-defined]
    trajectory_mod.format_for_sft = MagicMock(
        return_value=[{"role": "user", "content": "hello"}]
    )  # type: ignore[attr-defined]
    sys.modules["model_training.trajectory"] = trajectory_mod

    try:
        with patch("training_svc.storage.engine", test_engine):
            _run_hypernetwork_job(job_id, trajectory_id, task_type)

        # -- Verify job completed --
        job = JOB_STORE[job_id]
        assert job.status == "completed", (
            f"Expected status 'completed', got '{job.status}': {job.error}"
        )

        # -- Verify AdapterRecord stored in registry --
        registry = AdapterRegistry(engine=test_engine)
        record = registry.retrieve_by_id(adapter_id)

        assert record.source == "hypernetwork"
        assert record.rank == 8
        assert record.task_type == task_type
        assert record.session_id == trajectory_id
        assert len(record.file_hash) == 64, (
            "file_hash should be a 64-char SHA-256 hex string"
        )
        assert record.file_size_bytes > 0
    finally:
        # Clean up injected sys.modules to avoid contaminating other tests
        for mod_name in [
            "torch",
            "transformers",
            "model_training",
            "model_training.hypernetwork",
            "model_training.trajectory",
        ]:
            sys.modules.pop(mod_name, None)
