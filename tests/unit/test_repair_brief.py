"""Golden tests for deterministic repair brief classifiers (v2 session failures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rune.engine.repair_brief import RepairBrief, build_repair_brief

SESSIONS = Path("/tmp/goal3/rerun_failures2/sessions")
SIGNATURES = {
    "3754": "class Solution:\n    def maxDistance(self, s: str, k: int) -> int:\n        ",
    "3748": "class Solution:\n    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:\n        ",
    "3777": "class Solution:\n    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:\n        ",
    "3799": "class Solution:\n    def totalNumbers(self, digits: List[int]) -> int:\n        ",
    "3801": "class Solution:\n    def beautifulNumbers(self, l: int, r: int) -> int:\n        ",
    "3753": "class Solution:\n    def maxDifference(self, s: str) -> int:\n        ",
}
ENTRY = {
    "3754": "maxDistance",
    "3748": "sortMatrix",
    "3777": "maxProduct",
    "3799": "totalNumbers",
    "3801": "beautifulNumbers",
    "3753": "maxDifference",
}


def _stderr(qid: str, step: int) -> str:
    path = SESSIONS / qid / "session.jsonl"
    if not path.exists():
        raise pytest.skip.Exception(f"missing session {path}")
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["step"] == step:
            fb = rec.get("feedback") or {}
            return str(fb.get("stderr", ""))
    raise pytest.skip.Exception(f"step {step} not in {qid}")


@pytest.mark.parametrize(
    ("qid", "step", "failure_class", "replan", "invariant_substr"),
    [
        (
            "3753",
            2,
            "signature",
            False,
            "starter signature",
        ),
        (
            "3777",
            2,
            "complexity",
            False,
            "public examples",
        ),
        (
            "3801",
            2,
            "signature",
            False,
            "starter signature",
        ),
        (
            "3754",
            4,
            "arity",
            False,
            "parameters",
        ),
        (
            "3799",
            2,
            "import",
            False,
            "unimported",
        ),
        (
            "3748",
            2,
            "assertion",
            False,
            "anti-diagonal",
        ),
    ],
)
def test_v2_golden_brief(
    qid: str,
    step: int,
    failure_class: str,
    replan: bool,
    invariant_substr: str,
) -> None:
    stderr = _stderr(qid, step)
    brief = build_repair_brief(
        stderr,
        entry_point=ENTRY[qid],
        signature=SIGNATURES[qid],
    )
    assert brief is not None
    assert brief.failure_class == failure_class
    assert brief.replan_recommended is replan
    assert invariant_substr.lower() in brief.violated_invariant.lower()


def test_brief_serialization() -> None:
    b = RepairBrief(
        failure_class="arity",
        violated_invariant="x",
        observed="y",
        expected="z",
        fix_directive="fix",
        replan_recommended=False,
    )
    assert "failure_class: arity" in b.format_block()
    assert b.to_dict()["failure_class"] == "arity"
