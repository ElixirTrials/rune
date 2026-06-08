"""No-regress: ship the best candidate per subtask, never a later worse one.

Issue #52 RC-C: a re-code/repair after a near-miss can produce a crash; the
engine used to ship the LAST attempt (the crash), throwing away a would-be
success. The engine now retains the highest-quality candidate per subtask
(pass > assertion-mismatch > runtime crash > syntax/empty) and ships that.
"""

from __future__ import annotations

from dataclasses import replace

from rune.engine.parse import candidate_quality, parse_output
from rune.engine.policy import ACTIONS
from rune.engine.state import Feedback

_CONSTRAINT_SCALE_STDERR = (
    "Task requirements failed — fix exactly:\n"
    "- constraint_scale: measured O(n²) (Quadratic); Constraints allow n≤100000 "
    "— need O(n log n) or better"
)


def _passed() -> Feedback:
    return Feedback(stdout="", stderr="", exit_code=0)


def _mismatch() -> Feedback:
    return Feedback(stdout="", stderr="AssertionError: f(x) -> 1, want 2", exit_code=1)


def _crash() -> Feedback:
    return Feedback(stdout="", stderr="NameError: name 'q' is not defined", exit_code=1)


def _syntax() -> Feedback:
    return Feedback(stdout="", stderr="SyntaxError: invalid syntax", exit_code=1)


class TestCandidateQuality:
    def test_rank_order(self) -> None:
        assert candidate_quality("def f(): pass", _passed()) == 3
        assert candidate_quality("def f(): ...", _mismatch()) == 2
        assert candidate_quality("def f(): ...", _crash()) == 1
        assert candidate_quality("def f(:", _syntax()) == 0
        assert candidate_quality("", _passed()) == 0  # empty never ships

    def test_constraint_scale_only_ranks_as_visible_correct(self) -> None:
        fb = Feedback(stdout="", stderr=_CONSTRAINT_SCALE_STDERR, exit_code=1)
        assert candidate_quality("def f(): pass", fb) == 3
        assert (
            candidate_quality("def f(): pass", fb, constraint_scale_pass_quality=False)
            == 1
        )


class TestNoRegress:
    def _code_step(self, target: str, code: str, fb: Feedback, state: dict) -> dict:
        action = replace(ACTIONS["code"], target_subtask=target)
        return parse_output(action, "", fb, state, code=code)

    def test_crash_after_near_miss_keeps_the_near_miss(self) -> None:
        state: dict = {"subtasks": [], "code_results": {}, "code_passed": {}}
        # attempt 1: a near-miss (runs, asserts, mismatches)
        state.update(self._code_step("f", "def f(x): return 1", _mismatch(), state))
        # attempt 2: a re-code that crashes
        state.update(self._code_step("f", "def f(x): return q", _crash(), state))
        assert state["best_code"]["f"] == "def f(x): return 1"  # the near-miss
        assert state["best_quality"]["f"] == 2
        # last-attempt code_results still reflects the latest (for conditioning)
        assert state["code_results"]["f"] == "def f(x): return q"

    def test_pass_is_retained_over_a_later_failure(self) -> None:
        state: dict = {"subtasks": [], "code_results": {}, "code_passed": {}}
        state.update(self._code_step("f", "def f(x): return 2", _passed(), state))
        state.update(self._code_step("f", "def f(x): return 9", _crash(), state))
        assert state["best_code"]["f"] == "def f(x): return 2"
        assert state["best_quality"]["f"] == 3

    def test_improvement_is_taken(self) -> None:
        state: dict = {"subtasks": [], "code_results": {}, "code_passed": {}}
        state.update(self._code_step("f", "def f(x): return q", _crash(), state))
        state.update(self._code_step("f", "def f(x): return 2", _passed(), state))
        assert state["best_code"]["f"] == "def f(x): return 2"
        assert state["best_quality"]["f"] == 3

    def test_slow_but_correct_beats_later_assertion_fail(self) -> None:
        state: dict = {"subtasks": [], "code_results": {}, "code_passed": {}}
        slow = "def f(x): return 3"
        wrong = "def f(x): return 0"
        state.update(
            self._code_step(
                "f",
                slow,
                Feedback(stdout="", stderr=_CONSTRAINT_SCALE_STDERR, exit_code=1),
                state,
            )
        )
        state.update(self._code_step("f", wrong, _mismatch(), state))
        assert state["best_code"]["f"] == slow
        assert state["best_quality"]["f"] == 3
