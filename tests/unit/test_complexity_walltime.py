"""Walltime hardening: static-floor short-circuit + killable-subprocess timeout.

Both are CPU-only and self-contained (no /tmp/lcb data needed).
"""

from __future__ import annotations

import time

from rune.engine.complexity import (
    ComplexityProbeConfig,
    check_constraint_scale,
    check_constraint_scale_guarded,
)

# Constraints imply n up to 1e5 (allowed class ~ Linearithmic) while the public
# example uses a tiny list, so constraint_scale is required.
_SPEC = "Given nums, return a number.\n\nConstraints:\n1 <= nums.length <= 10^5\n"
_SIG = "def f(nums):"
_PUBLIC = "assert f([1, 2]) == 3"

_FAST = ComplexityProbeConfig(min_n=8, max_n=64, n_repeats=2, per_run_timeout_s=30.0)


def test_static_floor_short_circuits_without_running_big_o() -> None:
    # Combinatorial code: the static floor (Exponential) alone proves
    # infeasibility, so the gate must short-circuit with a 'static analysis'
    # verdict and NOT run big_o (which would execute the 2^n function).
    comb = """def f(nums):
    from itertools import combinations
    total = 0
    for r in range(len(nums) + 1):
        for c in combinations(nums, r):
            total += sum(c)
    return total
"""
    t0 = time.monotonic()
    out = check_constraint_scale(
        comb,
        entry_point="f",
        spec=_SPEC,
        public_checks=_PUBLIC,
        signature=_SIG,
        probe_config=_FAST,
    )
    elapsed = time.monotonic() - t0
    assert out.required and not out.ok
    assert "static analysis" in out.message
    # Short-circuit means big_o never ran the combinatorial function.
    assert elapsed < 2.0


def test_guarded_hard_kills_runaway_probe() -> None:
    # Exponential RECURSION has no loops, so the static floor can't see it and
    # big_o would run it to a hang. The killable subprocess must hard-kill and
    # return None within roughly the wall budget (not block on the dead thread).
    slow = """def f(nums):
    def rec(i):
        if i >= len(nums):
            return 0
        return rec(i + 1) + rec(i + 1)
    return rec(0)
"""
    t0 = time.monotonic()
    out = check_constraint_scale_guarded(
        slow,
        entry_point="f",
        spec=_SPEC,
        public_checks="assert f([1, 2]) == 0",
        signature=_SIG,
        probe_config=_FAST,
        wall_timeout_s=3.0,
    )
    elapsed = time.monotonic() - t0
    assert out is None
    assert elapsed < 20.0  # returned after the kill, did not wait on the thread


def test_guarded_returns_outcome_for_fast_linear() -> None:
    linear = "def f(nums):\n    return sum(nums)\n"
    out = check_constraint_scale_guarded(
        linear,
        entry_point="f",
        spec=_SPEC,
        public_checks=_PUBLIC,
        signature=_SIG,
        probe_config=_FAST,
        wall_timeout_s=30.0,
    )
    assert out is not None
    assert out.ok  # O(n) is feasible for n<=1e5
