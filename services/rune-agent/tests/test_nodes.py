"""Green-phase behavior tests for rune-agent node functions.

Tests verify the actual behavior of all 4 node functions:
generate_node, execute_node, reflect_node, save_trajectory_node.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inference import GenerationResult
from rune_agent.nodes import (
    execute_node,
    generate_node,
    reflect_node,
    save_trajectory_node,
)


def _make_generation_result(text: str) -> GenerationResult:
    """Helper to build a GenerationResult for mocking."""
    return GenerationResult(
        text=text,
        model="test-model",
        adapter_id=None,
        token_count=10,
        finish_reason="stop",
    )


def _mock_provider(text: str) -> MagicMock:
    """Return a mock InferenceProvider whose generate() returns the given text."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=_make_generation_result(text))
    return provider


# ---------------------------------------------------------------------------
# generate_node tests
# ---------------------------------------------------------------------------


async def test_generate_node_first_attempt() -> None:
    """generate_node first attempt: extracted code returned from python block."""
    code_in_block = "def fib(n):\n    return n"
    llm_response = f"```python\n{code_in_block}\n```"
    provider = _mock_provider(llm_response)

    state: dict[str, Any] = {
        "task_description": "Write fibonacci",
        "task_type": "function",
        "test_suite": "assert fib(1) == 1",
        "adapter_ids": [],
        "attempt_count": 0,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
    }

    with patch("rune_agent.nodes.get_provider", return_value=provider):
        result = await generate_node(state)

    assert result["generated_code"] == code_in_block
    provider.generate.assert_called_once()


async def test_generate_node_retry_prompt_includes_errors() -> None:
    """generate_node retry: prompt includes 'previous attempt' context."""
    provider = _mock_provider("```python\nprint('fixed')\n```")

    state: dict[str, Any] = {
        "task_description": "Write fibonacci",
        "task_type": "function",
        "test_suite": "assert fib(5) == 5",
        "adapter_ids": [],
        "attempt_count": 1,
        "generated_code": "def fib(n): return n",
        "stdout": "",
        "stderr": "AssertionError",
        "exit_code": 1,
    }

    with patch("rune_agent.nodes.get_provider", return_value=provider):
        await generate_node(state)

    call_args = provider.generate.call_args
    prompt_used = call_args.kwargs.get("prompt") or call_args.args[0]
    assert "previous attempt" in prompt_used.lower()


async def test_generate_node_no_code_block_fallback() -> None:
    """generate_node: full response used when no python block present."""
    raw_text = "def fib(n):\n    return n"
    provider = _mock_provider(raw_text)

    state: dict[str, Any] = {
        "task_description": "Write fibonacci",
        "task_type": "function",
        "test_suite": "assert fib(1) == 1",
        "adapter_ids": [],
        "attempt_count": 0,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
    }

    with patch("rune_agent.nodes.get_provider", return_value=provider):
        result = await generate_node(state)

    assert result["generated_code"] == raw_text.strip()


async def test_generate_node_with_adapter() -> None:
    """generate_node: adapter_ids[0] passed as adapter_id when non-empty."""
    provider = _mock_provider("```python\npass\n```")

    state: dict[str, Any] = {
        "task_description": "Write fibonacci",
        "task_type": "function",
        "test_suite": "assert True",
        "adapter_ids": ["my-adapter"],
        "attempt_count": 0,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
    }

    with patch("rune_agent.nodes.get_provider", return_value=provider):
        await generate_node(state)

    call_kwargs = provider.generate.call_args.kwargs
    assert call_kwargs.get("adapter_id") == "my-adapter"


# ---------------------------------------------------------------------------
# execute_node tests
# ---------------------------------------------------------------------------


async def test_execute_node_passing_script() -> None:
    """execute_node: simple passing script returns tests_passed=True, exit_code=0."""
    state: dict[str, Any] = {
        "generated_code": "x = 1",
        "test_suite": "assert x == 1",
    }
    result = await execute_node(state)

    assert result["tests_passed"] is True
    assert result["exit_code"] == 0
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)


async def test_execute_node_failing_script() -> None:
    """execute_node: failing assertion returns tests_passed=False, exit_code!=0."""
    state: dict[str, Any] = {
        "generated_code": "# intentionally empty",
        "test_suite": "assert False, 'expected failure'",
    }
    result = await execute_node(state)

    assert result["tests_passed"] is False
    assert result["exit_code"] != 0
    assert len(result["stderr"]) > 0


async def test_execute_node_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute_node: timeout returns error message in stderr, tests_passed=False."""
    monkeypatch.setenv("RUNE_EXEC_TIMEOUT", "1")

    state: dict[str, Any] = {
        "generated_code": "import time; time.sleep(60)",
        "test_suite": "",
    }
    result = await execute_node(state)

    assert result["tests_passed"] is False
    assert result["exit_code"] == 1
    assert "timed out" in result["stderr"].lower()


# ---------------------------------------------------------------------------
# reflect_node tests
# ---------------------------------------------------------------------------


async def test_reflect_node_increments_and_appends() -> None:
    """reflect_node: increments attempt_count and appends step to trajectory."""
    state: dict[str, Any] = {
        "attempt_count": 0,
        "generated_code": "def fib(n): pass",
        "stdout": "",
        "stderr": "error",
        "exit_code": 1,
        "tests_passed": False,
        "trajectory": [],
    }
    result = await reflect_node(state)

    assert result["attempt_count"] == 1
    assert len(result["trajectory"]) == 1
    step = result["trajectory"][0]
    assert step["generated_code"] == state["generated_code"]
    assert step["stdout"] == state["stdout"]
    assert step["stderr"] == state["stderr"]
    assert step["exit_code"] == state["exit_code"]
    assert step["tests_passed"] == state["tests_passed"]


async def test_reflect_node_immutable_trajectory() -> None:
    """reflect_node: original trajectory list is not mutated."""
    original_trajectory: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "attempt_count": 2,
        "generated_code": "x = 1",
        "stdout": "ok",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": True,
        "trajectory": original_trajectory,
    }
    result = await reflect_node(state)

    # Original list must remain empty -- no in-place mutation
    assert original_trajectory == []
    assert len(result["trajectory"]) == 1


# ---------------------------------------------------------------------------
# save_trajectory_node tests
# ---------------------------------------------------------------------------


async def test_save_trajectory_node_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """save_trajectory_node: outcome='success' when tests_passed=True."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))

    state: dict[str, Any] = {
        "session_id": "test-session-001",
        "tests_passed": True,
        "trajectory": [],
        "task_description": "Write fib",
        "task_type": "function",
        "adapter_ids": [],
    }

    with patch("rune_agent.nodes.record_trajectory") as mock_record:
        mock_record.return_value = {
            "session_id": "test-session-001",
            "file_path": "/tmp/x.json",
        }
        result = await save_trajectory_node(state)

    assert result["outcome"] == "success"
    mock_record.assert_called_once_with(
        session_id="test-session-001",
        steps=[],
        outcome="success",
        task_description="Write fib",
        task_type="function",
        adapter_ids=[],
    )


async def test_execute_node_uses_sandbox_backend() -> None:
    """execute_node delegates to the sandbox backend from get_sandbox_backend()."""
    mock_result = MagicMock()
    mock_result.stdout = "ok\n"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    mock_result.is_timed_out = False

    mock_backend = MagicMock()
    mock_backend.run.return_value = mock_result

    state: dict[str, Any] = {
        "generated_code": "x = 1",
        "test_suite": "assert x == 1",
    }

    with patch("rune_agent.nodes.get_sandbox_backend", return_value=mock_backend):
        result = await execute_node(state)

    mock_backend.run.assert_called_once()
    assert result["tests_passed"] is True
    assert result["stdout"] == "ok\n"


async def test_save_trajectory_node_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """save_trajectory_node: outcome='exhausted' when tests_passed=False."""
    monkeypatch.setenv("RUNE_TRAJECTORY_DIR", str(tmp_path))

    state: dict[str, Any] = {
        "session_id": "test-session-002",
        "tests_passed": False,
        "trajectory": [{"step": 1}],
        "task_description": "Write sort",
        "task_type": "function",
        "adapter_ids": ["adapter-1"],
    }

    with patch("rune_agent.nodes.record_trajectory") as mock_record:
        mock_record.return_value = {
            "session_id": "test-session-002",
            "file_path": "/tmp/y.json",
        }
        result = await save_trajectory_node(state)

    assert result["outcome"] == "exhausted"
    mock_record.assert_called_once_with(
        session_id="test-session-002",
        steps=[{"step": 1}],
        outcome="exhausted",
        task_description="Write sort",
        task_type="function",
        adapter_ids=["adapter-1"],
    )
