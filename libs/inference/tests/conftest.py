"""Pytest configuration for libs/inference.

Provides a mock vLLM client fixture for testing inference functions
without a running vLLM server.
"""

from typing import Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_vllm_client() -> Generator[MagicMock, None, None]:
    """Return a MagicMock standing in for the vLLM OpenAI client.

    The mock has .chat.completions.create pre-configured to return a stub
    response object with choices[0].message.content = "mock response".
    """
    client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "mock response"
    client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    yield client
