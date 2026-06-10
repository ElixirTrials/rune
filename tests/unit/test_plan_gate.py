"""Plan gate golden tests from v2 session plans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rune.engine.plan_gate import validate_plan

SESSIONS = Path("/tmp/goal3/rerun_failures2/sessions")
SIGNATURES = {
    "3754": "class Solution:\n    def maxDistance(self, s: str, k: int) -> int:\n        ",
    "3777": "class Solution:\n    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:\n        ",
    "3799": "class Solution:\n    def totalNumbers(self, digits: List[int]) -> int:\n        ",
    "3801": "class Solution:\n    def beautifulNumbers(self, l: int, r: int) -> int:\n        ",
    "3748": "class Solution:\n    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:\n        ",
    "3753": "class Solution:\n    def maxDifference(self, s: str) -> int:\n        ",
}
ENTRY = {
    "3754": "maxDistance",
    "3777": "maxProduct",
    "3799": "totalNumbers",
    "3801": "beautifulNumbers",
    "3748": "sortMatrix",
    "3753": "maxDifference",
}


def _plan(qid: str) -> str:
    path = SESSIONS / qid / "session.jsonl"
    if not path.exists():
        raise pytest.skip.Exception(f"missing {path}")
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["action"] == "plan":
            out = rec.get("output", "")
            if out.startswith("{"):
                data = json.loads(out)
                return str(data.get("plan", out))
            return out
    raise pytest.skip.Exception(f"no plan in {qid}")


def _task_snippet(qid: str) -> str:
    snippets = {
        "3777": "find a non-empty subsequence of nums",
        "3799": "You are given an array of digits",
    }
    return snippets.get(qid, "")


@pytest.mark.parametrize("qid", ["3754", "3777", "3799", "3801"])
def test_bad_plans_rejected(qid: str) -> None:
    result = validate_plan(
        _plan(qid),
        entry_point=ENTRY[qid],
        signature=SIGNATURES[qid],
        task_spec=_task_snippet(qid),
    )
    assert not result.ok
    assert result.deficiencies


def test_good_plan_3753_passes() -> None:
    result = validate_plan(
        _plan("3753"),
        entry_point=ENTRY["3753"],
        signature=SIGNATURES["3753"],
    )
    assert result.ok


def test_sort_matrix_plan_passes_gate() -> None:
    """3748 plan mentions right entry; algorithm error is not a plan-gate issue."""
    result = validate_plan(
        _plan("3748"),
        entry_point=ENTRY["3748"],
        signature=SIGNATURES["3748"],
    )
    assert result.ok
