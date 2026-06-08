"""Constraint-derived complexity oracle tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rune.bench.lcb import build_public_assert_checks
from rune.engine.complexity import (
    ComplexityProbeConfig,
    check_constraint_complexity,
    complexity_probe_required,
    parse_task_constraints,
)

_FAST_PROBE = ComplexityProbeConfig(
    min_n=3,
    max_n=32,
    n_repeats=2,
    per_run_timeout_s=5.0,
)

_LCB = Path("/tmp/lcb/test6.jsonl")

pytestmark = pytest.mark.skipif(
    not _LCB.exists(),
    reason="requires /tmp/lcb/test6.jsonl (LCB v6 data, not available in CI)",
)


def _row(qid: str) -> dict:
    for line in _LCB.read_text().splitlines():
        r = json.loads(line)
        if r["question_id"] == qid:
            return r
    raise KeyError(qid)


def _desc(qid: str) -> str:
    r = _row(qid)
    desc = r["question_content"]
    if r.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + r["starter_code"]
    return desc


def test_parse_constraints_extracts_length_and_range() -> None:
    spec = _desc("3777")
    c = parse_task_constraints(spec)
    assert c is not None
    assert c.length_max.get("nums") == 150
    spec2 = _desc("3801")
    c2 = parse_task_constraints(spec2)
    assert c2 is not None
    assert c2.range_upper.get("r") == 10**9


def test_probe_required_when_constraints_exceed_public() -> None:
    for qid in ("3777", "3801"):
        r = _row(qid)
        fn = json.loads(r["metadata"])["func_name"]
        public = build_public_assert_checks(r)
        assert complexity_probe_required(
            public, fn, _desc(qid), signature=r.get("starter_code", "")
        )


def test_probe_not_required_when_public_covers_scale() -> None:
    r = _row("3799")
    fn = json.loads(r["metadata"])["func_name"]
    public = build_public_assert_checks(r)
    assert not complexity_probe_required(
        public, fn, _desc("3799"), signature=r.get("starter_code", "")
    )


def test_combinations_fail_constraint_complexity() -> None:
    r = _row("3777")
    fn = json.loads(r["metadata"])["func_name"]
    public = build_public_assert_checks(r)
    comb = """def maxProduct(nums, k, limit):
    from itertools import combinations
    for r in range(1, len(nums) + 1):
        for subset in combinations(range(len(nums)), r):
            pass
    return -1
"""
    cx = check_constraint_complexity(
        comb,
        entry_point=fn,
        spec=_desc("3777"),
        public_checks=public,
        signature=r.get("starter_code", ""),
        probe_config=_FAST_PROBE,
    )
    assert cx.required
    assert not cx.ok
    assert "constraint_scale" in cx.message


def test_brute_range_fail_constraint_complexity() -> None:
    r = _row("3801")
    fn = json.loads(r["metadata"])["func_name"]
    public = build_public_assert_checks(r)
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
    cx = check_constraint_complexity(
        brute,
        entry_point=fn,
        spec=_desc("3801"),
        public_checks=public,
        signature=r.get("starter_code", ""),
        probe_config=_FAST_PROBE,
    )
    assert cx.required
    assert not cx.ok


def test_no_constraints_section_skips_probe() -> None:
    cx = check_constraint_complexity(
        "def f(): pass",
        entry_point="f",
        spec="Add two numbers.",
        public_checks="assert f() == 1",
    )
    assert not cx.required
    assert cx.ok
