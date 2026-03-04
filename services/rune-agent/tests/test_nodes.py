"""TDD tests for rune-agent node functions.

Each test constructs a minimal state dict with only the keys that node reads
-- documents dependencies between node functions and RuneState fields.
Node functions are stubs that raise NotImplementedError; tests assert the raise
and verify the error message contains the function name.
"""

import pytest
from rune_agent.nodes import (
    execute_node,
    generate_node,
    reflect_node,
    save_trajectory_node,
)


async def test_generate_node():
    """Test generate_node raises NotImplementedError with function name.

    State keys read by generate_node:
    - task_description: str
    - task_type: str
    - test_suite: str
    - adapter_ids: list[str]
    """
    state = {
        "task_description": "Write fibonacci",
        "task_type": "function",
        "test_suite": "assert fib(5) == 5",
        "adapter_ids": [],
    }
    with pytest.raises(NotImplementedError, match="generate_node"):
        await generate_node(state)


async def test_execute_node():
    """Test execute_node raises NotImplementedError with function name.

    State keys read by execute_node:
    - generated_code: str
    """
    state = {
        "generated_code": "def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)",
    }
    with pytest.raises(NotImplementedError, match="execute_node"):
        await execute_node(state)


async def test_reflect_node():
    """Test reflect_node raises NotImplementedError with function name.

    State keys read by reflect_node:
    - attempt_count: int
    - generated_code: str
    - exit_code: int
    - tests_passed: bool
    - trajectory: list[dict]
    """
    state = {
        "attempt_count": 0,
        "generated_code": "def fib(n): pass",
        "exit_code": 1,
        "tests_passed": False,
        "trajectory": [],
    }
    with pytest.raises(NotImplementedError, match="reflect_node"):
        await reflect_node(state)


async def test_save_trajectory_node():
    """Test save_trajectory_node raises NotImplementedError with function name.

    State keys read by save_trajectory_node:
    - tests_passed: bool
    """
    state = {
        "tests_passed": True,
    }
    with pytest.raises(NotImplementedError, match="save_trajectory_node"):
        await save_trajectory_node(state)
