"""Empirical complexity-class gate (big_o), runnable without LCB data."""

from __future__ import annotations

from rune.engine.complexity import (
    ComplexityProbeConfig,
    allowed_complexity_for_max_n,
    check_constraint_scale,
    measured_complexity_rank,
)

_SPEC_RANGE = """\
Constraints:
1 <= l <= r < 10^9
"""

_SPEC_LIST = """\
Constraints:
1 <= nums.length <= 150
"""


def test_allowed_complexity_tightens_with_max_n() -> None:
    assert allowed_complexity_for_max_n(10**9)[1] < allowed_complexity_for_max_n(100)[1]
    assert allowed_complexity_for_max_n(5_000)[0] == "O(n²)"
    assert allowed_complexity_for_max_n(10**9)[0] == "O(log n)"


def test_combinatorial_code_fails_class_gate() -> None:
    comb = """def maxProduct(nums, k, limit):
    from itertools import combinations
    for r in range(1, len(nums) + 1):
        for subset in combinations(range(len(nums)), r):
            pass
    return -1
"""
    public = "assert maxProduct([1, 2, 3], 1, 10) == -1"
    cfg = ComplexityProbeConfig(min_n=3, max_n=16, n_repeats=2, per_run_timeout_s=10.0)
    outcome = check_constraint_scale(
        comb,
        entry_point="maxProduct",
        spec=_SPEC_LIST,
        public_checks=public,
        signature="def maxProduct(nums, k, limit):",
        probe_config=cfg,
    )
    assert outcome.required
    assert not outcome.ok
    # Combinatorial structure trips the static floor, so the gate short-circuits
    # with a static verdict instead of running big_o on the 2^n function.
    assert "static analysis" in outcome.message
    assert "need" in outcome.message


def test_brute_range_fails_for_billion_bound() -> None:
    brute = """def beautifulNumbers(l, r):
    count = 0
    for i in range(l, r + 1):
        s = str(i)
        p = 1
        for d in s:
            p *= int(d)
        if p % sum(int(d) for d in s) == 0:
            count += 1
    return count
"""
    public = "assert beautifulNumbers(1, 5) == 1"
    cfg = ComplexityProbeConfig(min_n=20, max_n=300, n_repeats=2, per_run_timeout_s=10.0)
    outcome = check_constraint_scale(
        brute,
        entry_point="beautifulNumbers",
        spec=_SPEC_RANGE,
        public_checks=public,
        signature="def beautifulNumbers(l, r):",
        probe_config=cfg,
    )
    assert outcome.required
    assert not outcome.ok
    assert any(
        token in outcome.message
        for token in ("O(n)", "Linear", "O(n²)", "Quadratic", "static", ">=")
    )
    assert "10^9" in outcome.message or "1000000000" in outcome.message


def test_linear_code_passes_modest_bound() -> None:
    linear = """def sumNums(nums):
    return sum(nums)
"""
    public = "assert sumNums([1, 2]) == 3"
    cfg = ComplexityProbeConfig(min_n=10, max_n=200, n_repeats=2, per_run_timeout_s=10.0)
    outcome = check_constraint_scale(
        linear,
        entry_point="sumNums",
        spec=_SPEC_LIST,
        public_checks=public,
        signature="def sumNums(nums):",
        probe_config=cfg,
    )
    assert outcome.required
    assert outcome.ok


def test_measured_rank_orders_classes() -> None:
    class _Fake:
        pass

    class Linear(_Fake):
        pass

    class Quadratic(_Fake):
        pass

    assert measured_complexity_rank(Linear()) < measured_complexity_rank(Quadratic())
