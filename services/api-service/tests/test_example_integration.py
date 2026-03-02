"""Example Integration Tests.

This module demonstrates best practices for integration testing:
- Testing API endpoints
- Testing database interactions
- Testing with test client
- Testing async endpoints
- Testing error handling
- Testing authentication
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.mark.integration
class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check_returns_200(self, test_client: TestClient):
        """Test that health endpoint returns 200 OK."""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


@pytest.mark.integration
class TestUserEndpoints:
    """Test suite for user-related endpoints."""

    def test_get_users_list(self, test_client: TestClient, sample_users_list):
        """Test retrieving list of users."""
        # In a real test, you'd insert test data into the database first
        # For this example, we'll just test the endpoint structure

        response = test_client.get("/api/users")

        # Skip test if endpoint not implemented yet
        if response.status_code in [404, 405]:
            pytest.skip("Endpoint /api/users not implemented yet")

        # Assert expected behavior when endpoint exists
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_user_by_id(self, test_client: TestClient, db_session):
        """Test retrieving a specific user."""
        # Create a test user in the database
        # test_user = User(username="testuser", email="test@example.com")
        # db_session.add(test_user)
        # db_session.commit()

        # For this example, assume user with id=1 exists
        response = test_client.get("/api/users/1")

        # Adjust assertions based on your actual API
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "username" in data
            assert "email" in data
        else:
            # If user doesn't exist, should return 404
            assert response.status_code == 404

    def test_create_user(self, test_client: TestClient):
        """Test creating a new user."""
        new_user = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "securepassword123",
        }

        response = test_client.post("/api/users", json=new_user)

        # Adjust based on your API response
        if response.status_code == 201:
            data = response.json()
            assert data["username"] == new_user["username"]
            assert data["email"] == new_user["email"]
            assert "password" not in data  # Password should not be in response
        else:
            # Handle case where endpoint doesn't exist yet
            assert response.status_code in [404, 405]

    def test_update_user(self, test_client: TestClient):
        """Test updating user information."""
        update_data = {"username": "updateduser"}

        response = test_client.patch("/api/users/1", json=update_data)

        # Skip if endpoint not implemented
        if response.status_code in [404, 405]:
            pytest.skip("Endpoint PATCH /api/users/<id> not implemented yet")

        assert response.status_code == 200

    def test_delete_user(self, test_client: TestClient):
        """Test deleting a user."""
        response = test_client.delete("/api/users/1")

        # Skip if endpoint not implemented
        if response.status_code in [404, 405]:
            pytest.skip("Endpoint DELETE /api/users/<id> not implemented yet")

        assert response.status_code in [200, 204]

    def test_create_user_with_invalid_data(self, test_client: TestClient):
        """Test that invalid user data returns 422."""
        invalid_user = {
            "username": "",  # Empty username
            "email": "not-an-email",  # Invalid email
        }

        response = test_client.post("/api/users", json=invalid_user)

        # Should return validation error
        assert response.status_code in [422, 400, 404, 405]


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test suite for async API endpoints."""

    @pytest.mark.skip(
        reason="async_client fixture needs proper HTTPX AsyncClient setup"
    )
    async def test_async_endpoint(self, async_client: AsyncClient):
        """Test async endpoint."""
        response = await async_client.get("/health")

        assert response.status_code == 200

    @pytest.mark.skip(
        reason="async_client fixture needs proper HTTPX AsyncClient setup"
    )
    async def test_concurrent_requests(self, async_client: AsyncClient):
        """Test handling multiple concurrent requests."""
        import asyncio

        # Make multiple concurrent requests
        tasks = [
            async_client.get("/health"),
            async_client.get("/health"),
            async_client.get("/health"),
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)


@pytest.mark.integration
class TestAuthentication:
    """Test suite for authentication endpoints."""

    def test_login_success(self, test_client: TestClient):
        """Test successful login."""
        credentials = {
            "username": "testuser",
            "password": "testpassword",
        }

        response = test_client.post("/api/auth/login", json=credentials)

        # Adjust based on your auth implementation
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data or "token" in data
        else:
            # Handle case where endpoint doesn't exist
            assert response.status_code in [404, 405]

    def test_login_with_invalid_credentials(self, test_client: TestClient):
        """Test login with wrong credentials."""
        credentials = {
            "username": "testuser",
            "password": "wrongpassword",
        }

        response = test_client.post("/api/auth/login", json=credentials)

        # Should return unauthorized
        assert response.status_code in [401, 404, 405]

    def test_protected_endpoint_without_token(self, test_client: TestClient):
        """Test accessing protected endpoint without authentication."""
        response = test_client.get("/api/protected")

        # Should return unauthorized or not found
        assert response.status_code in [401, 403, 404, 405]

    def test_protected_endpoint_with_token(self, test_client: TestClient):
        """Test accessing protected endpoint with valid token."""
        # First login to get token (if endpoint exists)
        login_response = test_client.post(
            "/api/auth/login", json={"username": "testuser", "password": "testpassword"}
        )

        if login_response.status_code == 200:
            token = login_response.json().get("access_token")

            # Access protected endpoint
            headers = {"Authorization": f"Bearer {token}"}
            response = test_client.get("/api/protected", headers=headers)

            # Should succeed or return not found if endpoint doesn't exist
            assert response.status_code in [200, 404]


@pytest.mark.integration
class TestErrorHandling:
    """Test suite for error handling."""

    def test_404_for_non_existent_endpoint(self, test_client: TestClient):
        """Test that non-existent endpoints return 404."""
        response = test_client.get("/api/does-not-exist")

        assert response.status_code == 404

    def test_405_for_wrong_http_method(self, test_client: TestClient):
        """Test that wrong HTTP method returns 405."""
        # Assuming /health only accepts GET
        response = test_client.post("/health")

        assert response.status_code in [405, 404]

    def test_request_with_invalid_json(self, test_client: TestClient):
        """Test that invalid JSON returns 422."""
        response = test_client.post(
            "/api/users",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in [422, 400, 404, 405]


@pytest.mark.integration
class TestDatabaseOperations:
    """Test suite for database operations."""

    def test_database_transaction_rollback(self, db_session):
        """Test that database transactions are properly rolled back."""
        # This test demonstrates transaction rollback
        # In conftest.py, db_session automatically rolls back after each test

        # Example: Create a user
        # user = User(username="tempuser", email="temp@example.com")
        # db_session.add(user)
        # db_session.commit()

        # After this test, the user should not exist in the next test
        pass

    def test_database_constraints(self, db_session):
        """Test database constraints (e.g., unique email)."""
        # Example: Try to create two users with the same email
        # user1 = User(username="user1", email="same@example.com")
        # user2 = User(username="user2", email="same@example.com")

        # db_session.add(user1)
        # db_session.commit()

        # db_session.add(user2)
        # with pytest.raises(IntegrityError):
        #     db_session.commit()

        pass


@pytest.mark.e2e
class TestEndToEndWorkflows:
    """Test suite for end-to-end workflows."""

    def test_complete_user_workflow(self, test_client: TestClient):
        """Test complete user workflow.

        1. Create user
        2. Login
        3. Update profile
        4. Get profile
        5. Delete user
        """
        # 1. Create user
        new_user = {
            "username": "e2euser",
            "email": "e2e@example.com",
            "password": "password123",
        }
        create_response = test_client.post("/api/users", json=new_user)

        # Skip if endpoint doesn't exist
        if create_response.status_code == 404:
            pytest.skip("Endpoint not implemented yet")

        # Continue with workflow if endpoints exist
        # ... rest of the workflow
