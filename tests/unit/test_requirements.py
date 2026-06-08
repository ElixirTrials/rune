"""Task requirements oracle framework tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rune.bench.lcb import build_public_assert_checks
from rune.engine.complexity import constraint_scale_required
from rune.engine.requirements import (
    TASK_REQUIREMENTS,
    ConstraintScaleRequirement,
    EntryPointRequirement,
    ExecutableRequirement,
    PublicContractRequirement,
    RequirementContext,
    SignatureRequirement,
    evaluate_task_requirements,
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


def _ctx(qid: str) -> RequirementContext:
    r = _row(qid)
    fn = json.loads(r["metadata"])["func_name"]
    desc = r["question_content"]
    if r.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + r["starter_code"]
    return RequirementContext(
        entry_point=fn,
        signature=r.get("starter_code", ""),
        spec=desc,
        public_checks=build_public_assert_checks(r),
    )


def test_requirement_registry_is_extensible() -> None:
    kinds = {req.kind for req in TASK_REQUIREMENTS}
    assert kinds == {
        "entry_point",
        "executable",
        "signature",
        "contract",
        "constraint_scale",
    }


def test_executable_allows_typing_and_collections_names() -> None:
    # The probe injects the same names the official LCB starter code imports, so
    # idiomatic `List`/`Counter` usage must NOT be flagged (issue #52: this used
    # to NameError and burn repair budget on 3748/3777/3799).
    ctx = _ctx("3799")
    typed = """def totalNumbers(digits: List[int]) -> int:
    count = Counter(digits)
    return len(count)
"""
    ok, defs = evaluate_task_requirements(
        typed,
        ctx,
        requirements=(EntryPointRequirement(), ExecutableRequirement()),
    )
    assert ok, f"typing/collections names wrongly flagged: {defs}"


def test_executable_catches_genuine_load_failure() -> None:
    # A name genuinely absent from the standard preamble must still fail at
    # def-time (annotation evaluation), so real load errors are not masked.
    ctx = _ctx("3799")
    broken = "def totalNumbers(digits: NDArray) -> int:\n    return 0\n"
    ok, defs = evaluate_task_requirements(
        broken,
        ctx,
        requirements=(EntryPointRequirement(), ExecutableRequirement()),
    )
    assert not ok
    assert defs and "executable" in defs[0]


def test_signature_and_contract_still_apply() -> None:
    ctx = _ctx("3754")
    grid = "def maxDistance(grid):\n    return 0\n"
    ok, defs = evaluate_task_requirements(
        grid,
        ctx,
        requirements=(SignatureRequirement(), PublicContractRequirement()),
    )
    assert not ok
    assert any("signature" in d or "contract" in d for d in defs)


def test_constraint_scale_skipped_for_small_bound_tasks() -> None:
    ctx_small = _ctx("3799")
    assert not constraint_scale_required(
        ctx_small.public_checks,
        ctx_small.entry_point,
        ctx_small.spec,
        signature=ctx_small.signature,
    )


def test_constraint_scale_fails_exponential_code() -> None:
    ctx_big = _ctx("3777")
    comb = """def maxProduct(nums, k, limit):
    from itertools import combinations
    for r in range(1, len(nums) + 1):
        for subset in combinations(range(len(nums)), r):
            pass
    return -1
"""
    ok, defs = evaluate_task_requirements(
        comb,
        ctx_big,
        requirements=(ConstraintScaleRequirement(),),
    )
    assert not ok
    assert defs and "constraint_scale" in defs[0]
