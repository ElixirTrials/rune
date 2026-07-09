"""Capacity fixes (issue #52 trace review 2026-07-09): (1) concise-code
instruction keeps chain-of-thought out of code generation; (2) budget-aware
adapter conditioning so the hypernet's 2048-token encoder window never silently
drops ## Review Feedback (measured: i1 3754 s6-s10 conditioned on Task+Code
only — total failure-signal blackout)."""

from __future__ import annotations

from rune.engine.graph import render_training_format_trajectory, state_to_ctx
from rune.engine.parse import render_template
from rune.engine.policy import _with_target
from rune.engine.state import make_initial_state
from tests.unit.test_repair_context_fix import _state

_FEEDBACK = "AssertionError: maxDistance(*['NWSE', 1]) -> 2, want 3"
_BIG_CODE = (
    "def maxDistance(s, k):\n"
    + "    # analysis line\n" * 2500
    + "    return 0\n"
)


class TestConciseCodeInstruction:
    def test_flag_off_prompts_unchanged(self) -> None:
        ctx = state_to_ctx(_state(), _with_target("code", "maxDistance"))
        for name in ("prompt_code", "prompt_code_repair"):
            assert "reasoning" not in render_template(name, **ctx).lower()

    def test_flag_on_renders_instruction(self) -> None:
        ctx = state_to_ctx(
            _state(concise_code_instruction=True),
            _with_target("code", "maxDistance"),
        )
        for name in (
            "prompt_code",
            "prompt_code_repair",
            "prompt_episodic_code",
            "prompt_episodic_repair",
        ):
            out = render_template(name, **ctx)
            assert "do NOT explain your reasoning" in out, name

    def test_zeroshot_never_carries_instruction(self) -> None:
        # The zero-shot floor must stay byte-identical to the base arm.
        ctx = state_to_ctx(
            _state(concise_code_instruction=True),
            _with_target("code", "maxDistance"),
        )
        assert "reasoning" not in render_template("prompt_zeroshot", **ctx).lower()

    def test_threading(self) -> None:
        assert make_initial_state("t", 4)["concise_code_instruction"] is False
        state = make_initial_state(
            "t", 4, run_config={"concise_code_instruction": True}
        )
        assert state["concise_code_instruction"] is True


class TestAdapterCondBudget:
    def test_no_budget_is_legacy_byte_identical(self) -> None:
        legacy = (
            "## Task\nspec\n\n## Current Code\ncode\n\n## Review Feedback\nerr"
        )
        out = render_training_format_trajectory("spec", "code", "err")
        assert out == legacy

    def test_budget_keeps_feedback_over_code(self) -> None:
        out = render_training_format_trajectory(
            "task spec " * 50,
            _BIG_CODE,
            _FEEDBACK,
            char_budget=6800,
            entry_point="maxDistance",
        )
        assert len(out) <= 6800
        assert "## Review Feedback" in out
        assert "want 3" in out  # the payload survives — the whole point
        assert "## Task" in out
        assert "def maxDistance" in out  # code present, shrunk not dropped

    def test_budget_noop_when_small(self) -> None:
        small = render_training_format_trajectory(
            "spec", "def f():\n    return 1", _FEEDBACK,
            char_budget=6800, entry_point="f",
        )
        legacy = render_training_format_trajectory(
            "spec", "def f():\n    return 1", _FEEDBACK
        )
        assert small == legacy

    def test_budget_attempts_fill_remainder(self) -> None:
        attempts = [
            {"code": f"def maxDistance(s, k):\n    return {i}", "error": f"got {i}"}
            for i in range(3)
        ]
        out = render_training_format_trajectory(
            "spec",
            "def maxDistance(s, k):\n    return 9",
            _FEEDBACK,
            attempts=attempts,
            char_budget=6800,
            entry_point="maxDistance",
        )
        assert "## Previous Attempts" in out
        assert len(out) <= 6800

    def test_threading(self) -> None:
        assert make_initial_state("t", 4)["adapter_cond_budget_fix"] is False
        state = make_initial_state(
            "t", 4, run_config={"adapter_cond_budget_fix": True}
        )
        assert state["adapter_cond_budget_fix"] is True
