"""repair_context_fix flag: thin full-mode repair prompts carry the failure
signal, and history truncation keeps the assert payload (issue #52 root-cause
investigation 2026-07-09: brief suppressed diagnose but was never rendered;
err[:300] head-cut dropped got/want from every attempt block)."""

from __future__ import annotations

from typing import Any

from rune.engine.graph import _format_attempts, state_to_ctx
from rune.engine.parse import render_template
from rune.engine.policy import _with_target
from rune.engine.state import Feedback, StepRecord, Subtask, make_initial_state

_LONG_ERR = (
    "Traceback (most recent call last):\n"
    + '  File "<string>", line 8, in <module>\n' * 8
    + "    assert _oracle_got_0 == 3\n"
    + "AssertionError: maxDistance(*['NWSE', 1]) -> 2, want 3"
)


def _state(**overrides: Any) -> dict[str, Any]:
    rec = StepRecord(
        step=2,
        action_name="code",
        target_subtask="maxDistance",
        adapter_id=None,
        feedback=Feedback(stdout="", stderr=_LONG_ERR, exit_code=1),
        generated_code="def maxDistance(s, k):\n    pass",
    )
    base: dict[str, Any] = {
        "task": "spec " * 400,  # > _PROJECT_LABEL_CAP
        "subtasks": [Subtask("maxDistance", "d", [])],
        "plans": {"maxDistance": "p"},
        "code_results": {"maxDistance": "def maxDistance(s, k):\n    pass"},
        "code_passed": {"maxDistance": False},
        "retries": {"maxDistance": 1},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {"maxDistance": Feedback(stdout="", stderr=_LONG_ERR, exit_code=1)},
        "integration_feedback": None,
        "diagnosis": {},
        "repair_briefs": {"maxDistance": "failure_class: assertion\nexpected 3 got 2"},
        "actions": [],
        "trajectory": [rec],
        "step": 3,
        "budget_remaining": 20,
        "entry_point": "maxDistance",
        "signature": "def maxDistance(s: str, k: int) -> int:",
    }
    base.update(overrides)
    return base


class TestPromptRendering:
    def _render(self, ctx: dict[str, Any]) -> str:
        return render_template("prompt_code_repair", **ctx)

    def test_flag_off_prompt_unchanged(self) -> None:
        ctx = state_to_ctx(_state(), _with_target("repair", "maxDistance"))
        out = self._render(ctx)
        assert "Repair brief" not in out
        assert "Last failure" not in out
        assert "Diagnosis: " in out

    def test_flag_on_renders_brief_and_last_failure(self) -> None:
        ctx = state_to_ctx(
            _state(repair_context_fix=True), _with_target("repair", "maxDistance")
        )
        out = self._render(ctx)
        assert "failure_class: assertion" in out
        assert "-> 2, want 3" in out


class TestHistoryTruncation:
    def test_flag_off_head_cut_drops_payload(self) -> None:
        ctx = state_to_ctx(_state(), _with_target("repair", "maxDistance"))
        block = _format_attempts(ctx["code_trajectory"])
        assert "want 3" not in block  # documents the pre-fix loss

    def test_flag_on_tail_cut_keeps_payload(self) -> None:
        ctx = state_to_ctx(
            _state(repair_context_fix=True), _with_target("repair", "maxDistance")
        )
        block = _format_attempts(ctx["code_trajectory"])
        assert "-> 2, want 3" in block


class TestProjectLabel:
    def test_flag_off_hard_cut(self) -> None:
        ctx = state_to_ctx(_state(), _with_target("repair", "maxDistance"))
        assert len(ctx["project_label"]) == 1200

    def test_flag_on_boundary_cut_marked(self) -> None:
        task = ("line one of the spec\n" * 100)[:3000]
        ctx = state_to_ctx(
            _state(task=task, repair_context_fix=True),
            _with_target("repair", "maxDistance"),
        )
        label = ctx["project_label"]
        assert label.endswith("[spec truncated]")
        head = label.rsplit("\n", 1)[0]
        assert task.startswith(head)
        assert head.endswith("spec")  # cut at a line boundary, not mid-word


class TestConfigThreading:
    def test_default_off(self) -> None:
        state = make_initial_state("t", 4)
        assert state["repair_context_fix"] is False

    def test_run_config_on(self) -> None:
        state = make_initial_state("t", 4, run_config={"repair_context_fix": True})
        assert state["repair_context_fix"] is True
