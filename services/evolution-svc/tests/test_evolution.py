"""TDD tests for evolution-svc endpoint routes (red phase — stubs return 501)."""


def test_evaluate_adapter(test_client):
    """Test POST /evaluate returns evaluation metrics."""
    response = test_client.post(
        "/evaluate",
        json={"adapter_id": "a-1", "task_type": "code-gen"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "fitness_score" in data
    assert "pass_rate" in data


def test_evolve_adapters(test_client):
    """Test POST /evolve returns evolved adapter info."""
    response = test_client.post(
        "/evolve",
        json={"adapter_ids": ["a-1", "a-2"], "task_type": "code-gen"},
    )
    assert response.status_code == 200


def test_promote_adapter(test_client):
    """Test POST /promote returns promotion confirmation."""
    response = test_client.post(
        "/promote",
        json={"adapter_id": "a-1", "target_level": "domain"},
    )
    assert response.status_code == 200


def test_prune_adapter(test_client):
    """Test POST /prune returns pruning confirmation."""
    response = test_client.post(
        "/prune",
        json={"adapter_id": "a-1"},
    )
    assert response.status_code == 200
