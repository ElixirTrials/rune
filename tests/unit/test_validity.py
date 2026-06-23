"""Unit tests for benchmark solution validity gate."""

from __future__ import annotations

import json
from pathlib import Path

from rune.bench.lcb import build_public_assert_checks
from rune.engine.parse import parse_output
from rune.engine.requirements import (
    PublicContractRequirement,
    RequirementContext,
    SignatureRequirement,
)
from rune.engine.state import Action, make_initial_state
from rune.engine.validity import format_validity_feedback, validate_solution

# Hermetic fixtures vendored from the LCB-v6 escalate run (see
# tests/fixtures/lcb_engine_fixes); no 134MB test6.jsonl / ephemeral /tmp needed.
_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "lcb_engine_fixes"

_Q3754_SIG = (
    "class Solution:\n    def maxDistance(self, s: str, k: int) -> int:\n        \n"
)
_Q3754_PUBLIC = 'assert maxDistance("NWSE", 1) == 1'
_GRID = """def maxDistance(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                pass
    return 0
"""
_SK = """def maxDistance(s, k):
    x = y = 0
    for c in s:
        if c == "N":
            y += 1
    return abs(x) + abs(y)
"""
_WRONG_TOTAL = """def totalNumbers(n):
    for num in range(1, n + 1):
        if all(int(d) % 2 == 0 for d in str(num)):
            pass
    return 0
"""
_WRONG_TOTAL_PUBLIC = "assert totalNumbers([2, 4, 6, 8]) == 3"


def _ctx(entry: str, signature: str, public: str, spec: str = "") -> RequirementContext:
    return RequirementContext(
        entry_point=entry,
        signature=signature,
        spec=spec,
        public_checks=public,
    )


def test_signature_rejects_grid_param() -> None:
    ctx = _ctx("maxDistance", _Q3754_SIG, _Q3754_PUBLIC)
    out = SignatureRequirement().check(_GRID, ctx)
    assert not out.ok and "grid" in out.message


def test_contract_rejects_grid_arity() -> None:
    ctx = _ctx("maxDistance", _Q3754_SIG, _Q3754_PUBLIC)
    out = PublicContractRequirement().check(_GRID, ctx)
    assert not out.ok and "contract" in out.message


def test_contract_rejects_wrong_scalar_param() -> None:
    ctx = _ctx("totalNumbers", "", _WRONG_TOTAL_PUBLIC)
    out = PublicContractRequirement().check(_WRONG_TOTAL, ctx)
    assert not out.ok and "contract" in out.message


def test_sk_passes_signature_and_contract() -> None:
    ctx = _ctx("maxDistance", _Q3754_SIG, _Q3754_PUBLIC)
    assert SignatureRequirement().check(_SK, ctx).ok
    assert PublicContractRequirement().check(_SK, ctx).ok


def test_feedback_lists_deficiencies() -> None:
    msg = format_validity_feedback(("signature: bad", "contract: arity"))
    assert "signature: bad" in msg
    assert "contract: arity" in msg


def test_decompose_does_not_inject_goal_extras() -> None:
    row = next(
        json.loads(line)
        for line in (_FIXTURES / "rows.jsonl").read_text().splitlines()
        if json.loads(line)["question_id"] == "3754"
    )
    desc = row["question_content"]
    if row.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + row["starter_code"]
    public = build_public_assert_checks(row)
    state = make_initial_state(
        desc, 12, "maxDistance", row.get("starter_code", ""), public
    )
    raw = json.loads(
        (_FIXTURES / "sessions" / "3754" / "session.jsonl").read_text().splitlines()[0]
    )["output"]
    out = parse_output(
        Action(
            "decompose", "decompose", "prompt_decompose_concise", "", None, False, None
        ),
        raw,
        None,
        state,
    )
    assert "PLAN REQUIREMENT" not in out["subtasks"][0].description


def test_validate_grid_fails_signature_or_contract() -> None:
    vr = validate_solution(
        _GRID,
        entry_point="maxDistance",
        signature=_Q3754_SIG,
        spec="ignored",
        public_checks=_Q3754_PUBLIC,
    )
    assert not vr.ok
    assert any("signature" in d or "contract" in d for d in vr.deficiencies)
