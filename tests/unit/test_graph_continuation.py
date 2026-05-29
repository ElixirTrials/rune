"""Unit tests for the continuation sub-loop in step_node."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from rune.engine.graph import step_node
from rune.engine.state import Feedback, Subtask
from rune.model.adapter import AdapterResult
from rune.model.inference import GenerationResult


def _make_state(*, code: str = "", exit_code: int = 0) -> dict[str, Any]:
    subtask = Subtask(name="_main", description="Write a LinkedList", depends_on=[])
    fb = Feedback(stdout="", stderr="", exit_code=exit_code) if code else None
    return {
        "task": "Write a class LinkedList with methods append, prepend",
        "subtasks": [subtask],
        "plans": {"_main": "Write a LinkedList"},
        "code_results": {"_main": code} if code else {},
        "code_passed": {"_main": exit_code == 0} if code else {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {"_main": fb} if fb else {},
        "integration_feedback": None,
        "diagnosis": {},
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": 5,
    }


class TestStepNodeContinuation:
    def test_continuation_uses_generate_continuation(self) -> None:
        """Verify that truncated code triggers the prefill+continue path."""
        state = _make_state()
        truncated_json = json.dumps({"code": "class Node:\n    def __init__(self):\n"})
        initial_result = GenerationResult(
            text=truncated_json,
            thinking="",
            tokens_used=100,
            truncated=True,
        )
        cont_result = GenerationResult(
            text="        self.data = None\n\nclass LinkedList:\n    pass\n",
            thinking="",
            tokens_used=50,
            truncated=False,
        )
        model = MagicMock()
        model.generate_adapter.return_value = AdapterResult(
            adapter_id="test123",
            state_dict={},
        )
        model.hotswap_adapter = MagicMock()
        model.generate = AsyncMock(return_value=initial_result)
        model.generate_continuation = AsyncMock(return_value=cont_result)

        config = {
            "configurable": {
                "model": model,
                "run_config": {
                    "max_tokens": 512,
                    "cont_budget": 3,
                    "cont_multiplier": 1.5,
                    "no_repeat_ngram_size": 12,
                },
            },
        }

        with patch("rune.engine.graph.run_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                stdout="",
                stderr="",
                exit_code=0,
            )
            asyncio.run(step_node(state, config))

        model.generate_continuation.assert_called()
        call_kwargs = model.generate_continuation.call_args.kwargs
        assert "class Node:" in call_kwargs["assistant_prefix"]
        assert call_kwargs["system_prompt"].startswith("Output only Python code")

    def test_continuation_exits_early_on_valid_syntax(self) -> None:
        """Verify that the loop exits when accumulated code compiles."""
        state = _make_state()
        truncated_json = json.dumps({"code": "class Node:\n    pass\n"})
        initial_result = GenerationResult(
            text=truncated_json,
            thinking="",
            tokens_used=100,
            truncated=True,
        )
        cont_result = GenerationResult(
            text="\nclass LinkedList:\n    pass\n",
            thinking="",
            tokens_used=30,
            truncated=True,
        )
        model = MagicMock()
        model.generate_adapter.return_value = AdapterResult(
            adapter_id="test456",
            state_dict={},
        )
        model.hotswap_adapter = MagicMock()
        model.generate = AsyncMock(return_value=initial_result)
        model.generate_continuation = AsyncMock(return_value=cont_result)

        config = {
            "configurable": {
                "model": model,
                "run_config": {
                    "max_tokens": 512,
                    "cont_budget": 5,
                    "cont_multiplier": 1.5,
                    "no_repeat_ngram_size": 12,
                },
            },
        }

        with patch("rune.engine.graph.run_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                stdout="",
                stderr="",
                exit_code=0,
            )
            asyncio.run(step_node(state, config))

        assert model.generate_continuation.call_count == 1
