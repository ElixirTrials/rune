"""TDD red-phase tests for adapter endpoints.

These tests assert the expected 200/201 behavior of the adapter endpoints.
They FAIL in the current state because the stubs return 501.
"""

import pytest


@pytest.mark.xfail(reason="stub returns 501", strict=True)
def test_list_adapters(test_client):
    """Test GET /adapters returns list of adapters."""
    response = test_client.get("/adapters")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.xfail(reason="stub returns 501", strict=True)
def test_get_adapter(test_client):
    """Test GET /adapters/{id} returns single adapter."""
    response = test_client.get("/adapters/test-adapter-123")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data


@pytest.mark.xfail(reason="stub returns 501", strict=True)
def test_create_adapter(test_client, make_adapter_record):
    """Test POST /adapters creates a new adapter using factory fixture."""
    record = make_adapter_record()
    response = test_client.post("/adapters", json=record.model_dump())
    assert response.status_code in (200, 201)
    data = response.json()
    assert "id" in data
