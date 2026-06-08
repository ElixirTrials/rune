"""Smoke tests for the hard-task pass@1 root causes (issue #52, q3753 B4 trace).

Three engine defects let a correct LCB solution score pass@1=0:
  P1  the requirements oracle rejected the canonical ``class Solution`` form on a
      bogus signature/contract mismatch (it did not strip ``self`` like the probe
      and grader do), flipping a *passing* solution to failing.
  P2  the ship gate submitted structurally-valid but logic-failing near-misses
      (``best_quality == 2``) that never passed the public check.
  P3  the repair "tried and failed" summary surfaced only the (identical) error
      line, so repeated wrong approaches were invisible to the model.

Each test below fails on the pre-fix code and passes after the fix.
"""

from __future__ import annotations

from rune.bench.runner import BenchTask, _benchmark_shippable, resolve_shipped_code
from rune.engine.graph import _format_tried_and_failed
from rune.engine.requirements import (
    RequirementContext,
    evaluate_state_requirements,
    evaluate_task_requirements,
)

ENTRY = "maxDifference"
SIG = "def maxDifference(s: str) -> int:"
PUBLIC = "assert maxDifference('aaaaabbc') == 3"

# LCB-3349 correct logic: max(odd freq) - min(even freq). For 'aaaaabbc'
# (a:5, b:2, c:1) -> max odd 5 - min even 2 == 3.
CORRECT_CLASS = """class Solution:
    def maxDifference(self, s: str) -> int:
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        odd = [v for v in freq.values() if v % 2 == 1]
        even = [v for v in freq.values() if v % 2 == 0]
        return max(odd) - min(even)
"""

CORRECT_BARE = """def maxDifference(s: str) -> int:
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    odd = [v for v in freq.values() if v % 2 == 1]
    even = [v for v in freq.values() if v % 2 == 0]
    return max(odd) - min(even)
"""

# The exact 3753 step-2 trajectory code: correct odd/even algorithm, class form.
STEP2_CLASS = """class Solution:
    def maxDifference(self, s: str) -> int:
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        odd_freq = []
        even_freq = []
        for char, count in freq.items():
            if count % 2 == 0:
                even_freq.append(count)
            else:
                odd_freq.append(count)
        max_diff = 0
        for odd_count in odd_freq:
            for even_count in even_freq:
                max_diff = max(max_diff, odd_count - even_count)
        return max_diff
"""

# Structurally valid, runs, but wrong: max(all) - min(all) == 4 != 3 (the
# regression the engine ultimately shipped at step 11).
WRONG_BARE = """def maxDifference(s: str) -> int:
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    vals = list(freq.values())
    return max(vals) - min(vals)
"""


def _ctx() -> RequirementContext:
    return RequirementContext(
        entry_point=ENTRY, signature=SIG, spec="", public_checks=PUBLIC
    )


def _state() -> dict:
    return {
        "entry_point": ENTRY,
        "signature": SIG,
        "task": "",
        "public_checks": PUBLIC,
        "subtasks": [],
    }


def _task() -> BenchTask:
    return BenchTask(
        task_id="3753",
        description="maximum odd/even frequency difference",
        test_code="",
        entry_point=ENTRY,
        signature=SIG,
        public_checks=PUBLIC,
    )


# --- P1: requirements oracle must not flip a correct class-form solution ----


def test_p1_requirements_accept_correct_class_form() -> None:
    ok, deficiencies = evaluate_task_requirements(CORRECT_CLASS, _ctx())
    assert ok, f"correct class form was flipped to failing: {deficiencies}"


def test_p1_requirements_end_to_end_state_class_form() -> None:
    # Discriminator the per-requirement test would miss: the FULL requirement
    # chain (signature -> contract -> constraint_scale) must pass on the raw
    # class-form code, not just SignatureRequirement.
    ok, deficiencies = evaluate_state_requirements(_state(), STEP2_CLASS)
    assert ok, f"3753 step-2 (correct) still rejected by requirements: {deficiencies}"


def test_p1_requirements_still_reject_wrong_param_name() -> None:
    # The genuine param-name check must survive normalization: a bare function
    # with the wrong parameter name still fails.
    wrong_param = "def maxDifference(text: str) -> int:\n    return 3\n"
    ok, deficiencies = evaluate_task_requirements(wrong_param, _ctx())
    assert not ok
    assert any("signature" in d for d in deficiencies)


# --- P2: ship gate must only ship code that passes the public check ---------


def test_p2_shippable_accepts_correct() -> None:
    assert _benchmark_shippable(_task(), CORRECT_BARE, _task().description)
    assert _benchmark_shippable(_task(), CORRECT_CLASS, _task().description)


def test_p2_shippable_rejects_failed_public() -> None:
    # Structurally valid, runs, wrong answer -> must NOT ship.
    assert not _benchmark_shippable(_task(), WRONG_BARE, _task().description)


def test_p2_resolve_ships_correct_class_form() -> None:
    # The strongest mechanism check: the 3753 step-2 correct class-form
    # candidate must now be shipped, and the shipped code must pass the public
    # check (i.e. it would score pass@1=True).
    state = {
        "best_code": {ENTRY: STEP2_CLASS},
        "best_quality": {ENTRY: 3},
        "integrated_code": "",
        "code_results": {ENTRY: STEP2_CLASS},
    }
    shipped = resolve_shipped_code(state, _task(), spec=_task().description)
    assert shipped.strip(), "correct class-form solution shipped nothing"
    assert _benchmark_shippable(_task(), shipped, _task().description)


def test_p2_resolve_empty_on_failed_public() -> None:
    # A wrong near-miss kept as best_code must ship empty, not the wrong code.
    state = {
        "best_code": {ENTRY: WRONG_BARE},
        "best_quality": {ENTRY: 2},
        "integrated_code": "",
        "code_results": {ENTRY: WRONG_BARE},
    }
    shipped = resolve_shipped_code(state, _task(), spec=_task().description)
    assert shipped == "", f"shipped a known-wrong near-miss: {shipped!r}"


# --- P3: repair history must surface distinct code approaches ---------------


def test_p3_tried_and_failed_surfaces_distinct_approaches() -> None:
    # Two attempts whose ERROR line is identical but whose CODE differs: the
    # summary must show the distinct approaches so the model stops repeating.
    same_err = "AssertionError: maxDifference(*['aaaaabbc']) -> 1, want 3"
    trajectory = [
        {
            "step": 6,
            "action": "repair",
            "code": "def maxDifference(s):\n    return abs(odd_count - even_count)\n",
            "error": same_err,
            "passed": False,
        },
        {
            "step": 8,
            "action": "repair",
            "code": "def maxDifference(s):\n    return len(odd_freq) // 2\n",
            "error": same_err,
            "passed": False,
        },
    ]
    out = _format_tried_and_failed(trajectory)
    assert "abs(odd_count - even_count)" in out
    assert "len(odd_freq) // 2" in out
