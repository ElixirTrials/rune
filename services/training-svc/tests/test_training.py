"""TDD tests for training-svc endpoint routes (red phase — stubs return 501)."""

import pytest


@pytest.mark.xfail(reason="stub returns 501", strict=True)
def test_train_lora(test_client):
    """Test POST /train/lora returns training job info."""
    response = test_client.post(
        "/train/lora",
        json={"task_type": "code-gen", "rank": 64, "epochs": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


@pytest.mark.xfail(reason="stub returns 501", strict=True)
def test_train_hypernetwork(test_client):
    """Test POST /train/hypernetwork returns training job info."""
    response = test_client.post(
        "/train/hypernetwork",
        json={"task_type": "code-gen", "trajectory_ids": ["t-1", "t-2"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


@pytest.mark.xfail(reason="stub returns 501", strict=True)
def test_get_job_status(test_client):
    """Test GET /jobs/{job_id} returns job status."""
    response = test_client.get("/jobs/job-123")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "job_id" in data
