"""Tests for training-svc endpoint routes."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_job_store():
    """Clear JOB_STORE before and after each test to prevent cross-test contamination."""
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


@pytest.mark.xfail(reason="hypernetwork endpoint is Phase 22 stub", strict=True)
def test_train_hypernetwork_still_501(test_client):
    """POST /train/hypernetwork returns 501 (Phase 22 stub — xfail expected)."""
    response = test_client.post(
        "/train/hypernetwork",
        json={"task_type": "gen", "trajectory_ids": ["t-1"]},
    )
    assert response.status_code == 200
