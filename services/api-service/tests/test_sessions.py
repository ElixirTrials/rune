"""TDD red-phase tests for session endpoints.

These tests assert the expected 200/201 behavior of the session endpoints.
They FAIL in the current state because the stubs return 501.
"""


def test_list_sessions(test_client):
    """Test GET /sessions returns list of coding sessions."""
    response = test_client.get("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_session(test_client):
    """Test GET /sessions/{id} returns single coding session."""
    response = test_client.get("/sessions/test-session-456")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data


def test_create_session(test_client, make_coding_session):
    """Test POST /sessions creates a new coding session using factory fixture."""
    session = make_coding_session()
    response = test_client.post("/sessions", json=session.model_dump())
    assert response.status_code in (200, 201)
    data = response.json()
    assert "session_id" in data
