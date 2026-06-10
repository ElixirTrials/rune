"""Hybrid complexity oracle: empirical budget + adapter judge fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune.engine.complexity import (
    build_complexity_assessment_task,
    static_complexity_signals,
)
from rune.engine.graph import (
    _run_constraint_complexity_oracle,
    render_complexity_assessment_adapter,
)
from rune.engine.parse import ComplexityJudgeResult

_SPEC_RANGE = """\
Task: count beautiful numbers.
Constraints:
1 <= l <= r < 10^9
"""


def test_static_complexity_signals_detects_nested_loops() -> None:
    code = """def f(l, r):
    for i in range(l, r + 1):
        for j in range(i):
            pass
"""
    signals = static_complexity_signals(code)
    assert any("nested loops" in s for s in signals)
    assert any("range" in s for s in signals)


def test_build_complexity_assessment_task_includes_constraints() -> None:
    task = build_complexity_assessment_task(
        _SPEC_RANGE,
        "beautifulNumbers",
        signature="def beautifulNumbers(l, r):",
    )
    assert "Constraints:" in task
    assert "beautifulNumbers" in task
    assert "10^9" in task or "Constraints" in task


def test_render_complexity_assessment_adapter_format() -> None:
    state = {
        "task": _SPEC_RANGE,
        "entry_point": "beautifulNumbers",
        "signature": "def beautifulNumbers(l, r):",
    }
    traj = render_complexity_assessment_adapter(
        state,
        "def beautifulNumbers(l, r):\n    return 0\n",
    )
    assert "## Task" in traj
    assert "## Current Code" in traj
    assert "## Review Feedback" in traj
    assert "Constraints:" in traj


@pytest.mark.asyncio
async def test_empirical_timeout_uses_adapter_judge() -> None:
    state = {
        "task": _SPEC_RANGE,
        "entry_point": "beautifulNumbers",
        "signature": "def beautifulNumbers(l, r):",
        "public_checks": "assert beautifulNumbers(1, 5) == 1",
        "complexity_probe_min_n": 8,
        "complexity_probe_max_n": 400,
        "complexity_probe_n_repeats": 2,
        "complexity_probe_per_run_timeout_s": 5.0,
    }
    code = """def beautifulNumbers(l, r):
    for i in range(l, r + 1):
        pass
    return 0
"""
    model = MagicMock()
    model.generate_adapter = MagicMock(
        return_value=MagicMock(adapter_id="cx", state_dict={})
    )
    model.hotswap_adapter = MagicMock()
    model.generate = AsyncMock(
        return_value=MagicMock(
            text=ComplexityJudgeResult(
                reason="linear scan over range",
                measured_complexity="O(n)",
                sufficient=False,
            ).model_dump_json()
        )
    )
    run_config = {
        "complexity_empirical_timeout_s": 0.05,
        "complexity_judge_enabled": True,
        "adapter_scaling": 1.0,
        "complexity_judge_max_tokens": 128,
        "complexity_judge_temperature": 0.1,
    }

    # Simulate the empirical probe exceeding the wall budget: the guarded
    # subprocess returns None, so the engine escalates to the adapter judge.
    with patch("rune.engine.graph.check_constraint_scale_guarded", return_value=None):
        outcome = await _run_constraint_complexity_oracle(
            model, state, code, run_config
        )

    assert outcome is not None
    assert outcome.required
    assert not outcome.ok
    assert "constraint_scale:" in outcome.message
    assert "adapter analysis" in outcome.message
    model.generate_adapter.assert_called_once()
    model.generate.assert_awaited_once()
