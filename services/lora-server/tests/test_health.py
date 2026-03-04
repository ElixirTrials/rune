"""TDD tests for lora-server health.py check_vllm_ready function.

Tests use mocked HTTP since we cannot hit a real vLLM server in tests.
Imports directly from Python source via conftest.py sys.path setup.
"""

from unittest.mock import AsyncMock, patch

import pytest
from health import check_vllm_ready


@pytest.mark.asyncio
async def test_check_vllm_ready_returns_true_when_vllm_healthy():
    """Test check_vllm_ready returns True when vLLM responds with 200."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("health.httpx.AsyncClient", return_value=mock_client):
        result = await check_vllm_ready()
    assert result is True


@pytest.mark.asyncio
async def test_check_vllm_ready_returns_false_on_connection_error():
    """Test check_vllm_ready returns False when vLLM is unreachable."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("health.httpx.AsyncClient", return_value=mock_client):
        result = await check_vllm_ready()
    assert result is False
