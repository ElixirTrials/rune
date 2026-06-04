"""In-loop model-judge: grounded correctness verdict that can flip code to failing."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from rune.engine.graph import _run_model_judge
from rune.engine.parse import JudgeResult


def _model(text: str) -> MagicMock:
    m = MagicMock()
    m.generate = AsyncMock(return_value=MagicMock(text=text))
    return m


class TestJudgeResult:
    def test_defaults_when_only_correct_present(self) -> None:
        v = JudgeResult.model_validate_json('{"correct": true}')
        assert v.correct is True
        assert v.failing_input == ""
        assert v.reason == ""

    def test_grounded_incorrect_verdict(self) -> None:
        v = JudgeResult.model_validate_json(
            '{"correct": false, "failing_input": "100/10/2", "reason": "int div"}'
        )
        assert v.correct is False
        assert v.failing_input == "100/10/2"


class TestRunModelJudge:
    def test_parses_incorrect_verdict(self) -> None:
        m = _model('{"correct": false, "failing_input": "100/10/2", "reason": "x"}')
        v = asyncio.run(_run_model_judge(m, "spec", "calculate", "code", {}))
        assert v is not None
        assert v.correct is False
        assert v.failing_input == "100/10/2"

    def test_fail_open_on_unparseable_output(self) -> None:
        # A flaky judge must not block already-passing code -> None (treated correct).
        m = _model("not json at all")
        v = asyncio.run(_run_model_judge(m, "spec", "f", "code", {}))
        assert v is None
