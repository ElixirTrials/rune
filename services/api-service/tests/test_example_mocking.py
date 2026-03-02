"""Example Tests with Mocking.

This module demonstrates best practices for mocking:
- Mocking external HTTP calls
- Mocking async HTTP calls
- Mocking environment variables
- Mocking file system operations
- Mocking time-dependent functions
"""

from datetime import datetime
from unittest.mock import Mock, mock_open, patch

import aiohttp
import pytest


class TestHTTPMocking:
    """Test suite demonstrating HTTP mocking."""

    @pytest.mark.skip(reason="Requires pytest-requests-mock: pip install requests-mock")
    def test_mock_requests_library(self, requests_mock):
        """Test mocking synchronous HTTP calls with requests-mock."""
        import requests

        # Mock the external API
        requests_mock.get(
            "https://api.example.com/data",
            json={"status": "success", "data": [1, 2, 3]},
            status_code=200,
        )

        # Make the request
        response = requests.get("https://api.example.com/data")

        # Verify
        assert response.status_code == 200
        assert response.json() == {"status": "success", "data": [1, 2, 3]}

    @pytest.mark.skip(reason="Requires pytest-requests-mock: pip install requests-mock")
    def test_mock_multiple_endpoints(self, requests_mock):
        """Test mocking multiple endpoints."""
        import requests

        # Mock multiple endpoints
        requests_mock.get(
            "https://api.example.com/users/1", json={"id": 1, "name": "User 1"}
        )
        requests_mock.get(
            "https://api.example.com/users/2", json={"id": 2, "name": "User 2"}
        )
        requests_mock.post(
            "https://api.example.com/users",
            json={"id": 3, "name": "New User"},
            status_code=201,
        )

        # Test the mocked endpoints
        user1 = requests.get("https://api.example.com/users/1").json()
        user2 = requests.get("https://api.example.com/users/2").json()
        new_user = requests.post(
            "https://api.example.com/users", json={"name": "New User"}
        )

        assert user1["name"] == "User 1"
        assert user2["name"] == "User 2"
        assert new_user.status_code == 201

    @pytest.mark.skip(reason="Requires pytest-requests-mock: pip install requests-mock")
    def test_mock_error_response(self, requests_mock):
        """Test mocking error responses."""
        import requests

        requests_mock.get(
            "https://api.example.com/error",
            status_code=500,
            json={"error": "Internal server error"},
        )

        response = requests.get("https://api.example.com/error")

        assert response.status_code == 500
        assert "error" in response.json()


class TestAsyncHTTPMocking:
    """Test suite for mocking async HTTP calls."""

    @pytest.mark.skip(reason="aioresponses fixture needs proper configuration")
    @pytest.mark.asyncio
    async def test_mock_aiohttp(self, aioresponses):
        """Test mocking async HTTP calls with aioresponses."""
        # Mock the async HTTP call
        aioresponses.get(
            "https://api.example.com/async-data",
            payload={"status": "success", "value": 42},
            status=200,
        )

        # Make the async request
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com/async-data") as response:
                data = await response.json()

        assert data == {"status": "success", "value": 42}

    @pytest.mark.skip(reason="aioresponses fixture needs proper configuration")
    @pytest.mark.asyncio
    async def test_mock_multiple_async_calls(self, aioresponses):
        """Test mocking multiple async calls."""
        aioresponses.get("https://api.example.com/data/1", payload={"id": 1})
        aioresponses.get("https://api.example.com/data/2", payload={"id": 2})

        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com/data/1") as resp1:
                data1 = await resp1.json()
            async with session.get("https://api.example.com/data/2") as resp2:
                data2 = await resp2.json()

        assert data1["id"] == 1
        assert data2["id"] == 2


class TestEnvironmentVariables:
    """Test suite for mocking environment variables."""

    def test_with_monkeypatch(self, monkeypatch):
        """Test mocking environment variables with monkeypatch."""
        import os

        # Set environment variable
        monkeypatch.setenv("API_KEY", "test-key-123")
        monkeypatch.setenv("MODEL_BACKEND", "vertex")

        # Verify
        assert os.getenv("API_KEY") == "test-key-123"
        assert os.getenv("MODEL_BACKEND") == "vertex"

    def test_delete_env_var(self, monkeypatch):
        """Test deleting environment variables."""
        import os

        # Ensure variable exists
        monkeypatch.setenv("TEMP_VAR", "value")
        assert os.getenv("TEMP_VAR") == "value"

        # Delete it
        monkeypatch.delenv("TEMP_VAR")
        assert os.getenv("TEMP_VAR") is None

    @patch.dict("os.environ", {"DATABASE_URL": "sqlite:///:memory:"})
    def test_with_patch_dict(self):
        """Test mocking env vars with patch.dict."""
        import os

        assert os.getenv("DATABASE_URL") == "sqlite:///:memory:"


class TestFileSystemMocking:
    """Test suite for mocking file system operations."""

    def test_mock_file_read(self):
        """Test mocking file read operations."""
        mock_data = "This is test file content\nLine 2\nLine 3"

        with patch("builtins.open", mock_open(read_data=mock_data)):
            with open("test.txt", "r") as f:
                content = f.read()

        assert content == mock_data

    def test_mock_file_write(self):
        """Test mocking file write operations."""
        m = mock_open()

        with patch("builtins.open", m):
            with open("test.txt", "w") as f:
                f.write("test content")

        # Verify write was called
        m.assert_called_once_with("test.txt", "w")
        handle = m()
        handle.write.assert_called_once_with("test content")

    def test_mock_file_exists(self, monkeypatch):
        """Test mocking file existence checks."""
        import os.path

        # Mock file exists
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        assert os.path.exists("/fake/path/file.txt")

        # Mock file doesn't exist
        monkeypatch.setattr(os.path, "exists", lambda x: False)
        assert not os.path.exists("/real/path/file.txt")


class TestTimeDependentFunctions:
    """Test suite for mocking time-dependent functions."""

    @patch("time.time")
    def test_mock_time(self, mock_time):
        """Test mocking time.time()."""
        import time

        # Mock current time
        mock_time.return_value = 1234567890.0

        assert time.time() == 1234567890.0

    @pytest.mark.skip(reason="Example needs freezegun or proper datetime mocking")
    @patch("datetime.datetime")
    def test_mock_datetime(self, mock_datetime):
        """Test mocking datetime.now()."""
        # Mock current datetime
        mock_now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        result = datetime.now()
        assert result == mock_now

    def test_freeze_time(self):
        """Test freezing time (requires freezegun library)."""
        # Example using freezegun (install with: pip install freezegun)
        # from freezegun import freeze_time
        #
        # with freeze_time("2024-01-01 12:00:00"):
        #     now = datetime.now()
        #     assert now.year == 2024
        #     assert now.month == 1
        #     assert now.hour == 12
        pass


class TestDependencyMocking:
    """Test suite for mocking class dependencies."""

    def test_mock_database_connection(self):
        """Test mocking database connection."""
        mock_db = Mock()
        mock_db.execute.return_value = [{"id": 1, "name": "Test"}]

        # Use the mock in your code
        result = mock_db.execute("SELECT * FROM users")

        assert len(result) == 1
        assert result[0]["name"] == "Test"
        mock_db.execute.assert_called_once()

    def test_mock_with_side_effect(self):
        """Test mocking with side effects."""
        mock_func = Mock(side_effect=[1, 2, 3])

        # Each call returns next value in sequence
        assert mock_func() == 1
        assert mock_func() == 2
        assert mock_func() == 3

    def test_mock_raises_exception(self):
        """Test mocking function that raises exception."""
        mock_func = Mock(side_effect=ValueError("Test error"))

        with pytest.raises(ValueError, match="Test error"):
            mock_func()


class TestModelInferenceMocking:
    """Test suite for mocking AI model inference."""

    @pytest.mark.skip(reason="Example - requires actual inference.factory module")
    @patch("inference.loaders.get_llm")
    def test_mock_llm_response(self, mock_get_llm):
        """Test mocking LLM response."""
        # Mock the LLM instance
        mock_llm = Mock()
        mock_llm.invoke.return_value = "This is a mocked LLM response"
        mock_get_llm.return_value = mock_llm

        # Use in your code
        from inference.loaders import get_llm

        llm = get_llm()
        response = llm.invoke("test prompt")

        assert response == "This is a mocked LLM response"
        mock_llm.invoke.assert_called_once_with("test prompt")

    @pytest.mark.skip(reason="Example - requires actual inference.factory module")
    @patch("inference.loaders.get_llm")
    async def test_mock_async_llm(self, mock_get_llm):
        """Test mocking async LLM calls."""
        from unittest.mock import AsyncMock

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "Async mocked response"
        mock_get_llm.return_value = mock_llm

        # Use in your code
        from inference.loaders import get_llm

        llm = get_llm()
        response = await llm.ainvoke("test prompt")

        assert response == "Async mocked response"


class TestCachingMocking:
    """Test suite for mocking caching behavior."""

    def test_mock_cache_hit(self):
        """Test mocking cache hit."""
        mock_cache = Mock()
        mock_cache.get.return_value = {"cached": "data"}

        result = mock_cache.get("key")

        assert result == {"cached": "data"}

    def test_mock_cache_miss(self):
        """Test mocking cache miss."""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        result = mock_cache.get("nonexistent_key")

        assert result is None
