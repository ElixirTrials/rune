"""TDD tests for lora-server vllm_client.py VLLMClient methods.

Tests assert NotImplementedError is raised for stub methods.
Imports directly from Python source via conftest.py sys.path setup.
"""

import pytest
from vllm_client import VLLMClient


@pytest.mark.asyncio
async def test_load_adapter_raises_not_implemented():
    """Test VLLMClient.load_adapter raises NotImplementedError."""
    client = VLLMClient()
    with pytest.raises(NotImplementedError, match="load_adapter"):
        await client.load_adapter("adapter-1", "Qwen/Qwen2.5-Coder-7B-Instruct")


@pytest.mark.asyncio
async def test_generate_raises_not_implemented():
    """Test VLLMClient.generate raises NotImplementedError."""
    client = VLLMClient()
    with pytest.raises(NotImplementedError, match="generate"):
        await client.generate("Write hello world", "Qwen/Qwen2.5-Coder-7B-Instruct")
