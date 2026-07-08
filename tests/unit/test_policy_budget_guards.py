"""Flag-gated budget guards in select_action (issue #52 §4 levers 3a/3b).

Both guards DEFAULT OFF: with the flag unset, action selection must be identical
to the pre-guard behaviour on the same scenario.
"""

from __future__ import annotations

from typing import Any

from rune.engine.policy import (
    _normalize_stderr,
    select_action,
)
from rune.engine.state import Feedback, StepRecord, Subtask

_COMPLEXITY_STDERR = (
    "Task requirements failed:\n"
    "- constraint_scale: static analysis indicates O(2^n); "
    "Constraints allow n<=150 - need O(n^3) or better"
)


def _rec(
    step: int, name: str, code: str, stderr: str, *, exit_code: int = 1
) -> StepRecord:
    return StepRecord(
        step=step,
        action_name="repair",
        target_subtask=name,
        adapter_id=None,
        feedback=Feedback(stdout="", stderr=stderr, exit_code=exit_code),
        generated_code=code,
    )


def _make_state(
    *, trajectory: list[StepRecord], **overrides: Any
) -> dict[str, Any]:
    subtasks = [Subtask("a", "do a", [])]
    fb = Feedback(stdout="", stderr="err", exit_code=1)
    base: dict[str, Any] = {
        "task": "test",
        "subtasks": subtasks,
        "plans": {"a": "plan"},
        "code_results": {"a": "bad"},
        "code_passed": {"a": False},
        "retries": {"a": 1},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {"a": fb},
        "integration_feedback": None,
        "diagnosis": {"a": "d"},
        "actions": [],
        "trajectory": trajectory,
        "step": len(trajectory),
        "budget_remaining": 20,
    }
    base.update(overrides)
    return base


class TestNormalizeStderr:
    def test_masks_volatile_substrings(self) -> None:
        a = _normalize_stderr(
            'File "/tmp/abc123/sol.py", line 7, at 0xdeadbeef took 0.42 s'
        )
        b = _normalize_stderr(
            'File "/tmp/zzz999/sol.py", line 42, at 0xfeedface took 9.99 s'
        )
        assert a == b

    def test_preserves_assertion_identity(self) -> None:
        a = _normalize_stderr("AssertionError: f(*[1]) -> 3, want 5")
        b = _normalize_stderr("AssertionError: f(*[1]) -> 4, want 5")
        assert a != b


class TestDedupGuard:
    def _traj_same(self) -> list[StepRecord]:
        # Three identical brute-force resubmissions, same stderr + return line.
        code = "def a():\n    return brute()"
        err = "AssertionError: a() -> 1, want 2"
        return [_rec(i, "a", code, err) for i in range(3)]

    def test_flag_off_is_default_behaviour(self) -> None:
        state = _make_state(trajectory=self._traj_same())  # no flag
        actions = select_action(state)
        assert len(actions) == 1
        assert actions[0].name in {"code", "repair", "diagnose"}
        assert actions[0].target_subtask == "a"

    def test_fires_after_n_identical(self) -> None:
        state = _make_state(trajectory=self._traj_same(), repair_dedup_after=3)
        # Guarded subtask is dropped from actions -> single-subtask exhaustion
        # ship-best path returns [] (best_code + public_checks present).
        state["entry_point"] = "a"
        state["public_checks"] = "assert a() == 2"
        state["best_code"] = {"a": "def a():\n    return brute()"}
        assert select_action(state) == []

    def test_does_not_fire_when_approach_changes(self) -> None:
        traj = self._traj_same()
        # Newest attempt uses a structurally different approach (return line).
        traj[-1] = _rec(
            2, "a", "def a():\n    return smart()", "AssertionError: a() -> 1, want 2"
        )
        state = _make_state(trajectory=traj, repair_dedup_after=3)
        actions = select_action(state)
        assert actions and actions[0].target_subtask == "a"

    def test_does_not_fire_when_stderr_changes(self) -> None:
        traj = self._traj_same()
        traj[-1] = _rec(
            2, "a", "def a():\n    return brute()", "AssertionError: a() -> 9, want 2"
        )
        state = _make_state(trajectory=traj, repair_dedup_after=3)
        actions = select_action(state)
        assert actions and actions[0].target_subtask == "a"

    def test_does_not_fire_below_threshold(self) -> None:
        state = _make_state(trajectory=self._traj_same(), repair_dedup_after=4)
        actions = select_action(state)
        assert actions and actions[0].target_subtask == "a"

    def test_n1_floored_to_2(self) -> None:
        # A window of 1 is trivially all-equal; n=1 must not kill the first
        # repair (it would have cut off 3799's genuine diagnose→repair chain).
        state = _make_state(
            trajectory=self._traj_same()[:1], repair_dedup_after=1
        )
        actions = select_action(state)
        assert actions and actions[0].target_subtask == "a"
        # Two identical failures: the floored window (2) fires.
        state = _make_state(
            trajectory=self._traj_same()[:2], repair_dedup_after=1
        )
        state["entry_point"] = "a"
        state["public_checks"] = "assert a() == 2"
        state["best_code"] = {"a": "def a():\n    return brute()"}
        assert select_action(state) == []


class TestComplexityCapGuard:
    def _traj_complexity(self, n: int) -> list[StepRecord]:
        return [
            _rec(i, "a", f"def a():\n    return brute_{i}()", _COMPLEXITY_STDERR)
            for i in range(n)
        ]

    def test_flag_off_is_default_behaviour(self) -> None:
        state = _make_state(trajectory=self._traj_complexity(3))
        actions = select_action(state)
        assert actions and actions[0].target_subtask == "a"

    def test_fires_after_k_complexity_only(self) -> None:
        state = _make_state(
            trajectory=self._traj_complexity(2), complexity_repair_cap=2
        )
        state["entry_point"] = "a"
        state["public_checks"] = "assert a() == 2"
        state["best_code"] = {"a": "def a():\n    return brute_0()"}
        assert select_action(state) == []

    def test_does_not_fire_on_non_complexity_failure(self) -> None:
        traj = self._traj_complexity(2)
        # Latest failure is a plain assertion, not complexity-only.
        traj[-1] = _rec(1, "a", "def a():\n    return x()", "AssertionError: nope")
        state = _make_state(trajectory=traj, complexity_repair_cap=2)
        actions = select_action(state)
        assert actions and actions[0].target_subtask == "a"

    def test_does_not_fire_below_cap(self) -> None:
        state = _make_state(
            trajectory=self._traj_complexity(1), complexity_repair_cap=2
        )
        actions = select_action(state)
        assert actions and actions[0].target_subtask == "a"
