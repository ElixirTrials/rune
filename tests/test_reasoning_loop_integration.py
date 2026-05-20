"""Integration tests for run_reasoning_loop in rune_runner."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_run_reasoning_loop_returns_result():
    """Verify run_reasoning_loop wraps the graph invocation correctly."""
    from scripts.rune_runner import run_reasoning_loop

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "generated_code": "def main(): pass",
        "tests_passed": True,
        "turn_count": 3,
        "turn_history": [{"turn": 0}, {"turn": 1}, {"turn": 2}],
        "artifact": None,
        "finish_reason": "stop",
    })

    result = await run_reasoning_loop(
        graph=mock_graph,
        initial_output="partial code...",
        task_description="Build a library",
        phase="code",
        phase_executes=True,
        max_turns=20,
        sliding_window_tokens=1024,
        scaling_factor=0.16,
        session_id="test-session",
    )

    assert result["generated_code"] == "def main(): pass"
    assert result["tests_passed"] is True
    mock_graph.ainvoke.assert_called_once()
