"""RC-targeted smoke tests for LCB oracle unification (fail-then-pass on pre-fix HEAD)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rune.bench.lcb import build_public_assert_checks
from rune.bench.runner import BenchTask, resolve_shipped_code, run_benchmark
from rune.engine.continuation import strip_self_tests
from rune.engine.graph import (
    apply_oracle_fail_closed,
    build_code_probe,
    resolve_in_loop_check,
)
from rune.engine.oracle import build_probe, extract_public_checks
from rune.engine.state import Feedback
from rune.sandbox.executor import run_in_sandbox

_LCB_ROW: dict[str, Any] = {
    "question_id": "3753",
    "question_content": (
        "You are given a string s consisting of lowercase English letters. "
        "Your task is to find the maximum difference between the frequency of "
        "two characters in the string such that:\n\n"
        "One of the characters has an even frequency in the string.\n"
        "The other character has an odd frequency in the string.\n\n"
        "Return the maximum difference, calculated as the frequency of the "
        "character with an odd frequency minus the frequency of the character "
        "with an even frequency.\n"
        "Example 1:\n\nInput: s = \"aaaaabbc\"\nOutput: 3\n\n"
        "Example 2:\n\nInput: s = \"abcabcab\"\nOutput: 1\n"
    ),
    "starter_code": (
        "class Solution:\n"
        "    def maxDifference(self, s: str) -> int:\n"
        "        \n"
    ),
    "public_test_cases": json.dumps(
        [
            {"input": '"aaaaabbc"', "output": "3"},
            {"input": '"abcabcab"', "output": "1"},
        ]
    ),
    "metadata": json.dumps({"func_name": "maxDifference"}),
}

CORRECT_MAXDIFF = """\
def maxDifference(s):
    from collections import Counter
    c = Counter(s)
    odds = [v for v in c.values() if v % 2]
    evens = [v for v in c.values() if v % 2 == 0]
    return max(o - e for o in odds for e in evens)
"""

WRONG_MAXMIN = """\
def maxDifference(s):
    from collections import Counter
    c = Counter(s)
    return max(c.values()) - min(c.values())
"""

MODEL_ACCEPTANCE = (
    "assert maxDifference('aaaaabbc') == 3\n"
    "assert maxDifference('abcabcab') == 1\n"
    "assert maxDifference('aabbcc') == 0\n"
    "assert maxDifference('abcdef') == 5"
)

_LEGACY_SOLUTION_TEST = """\
from typing import List
_s = Solution()
assert _s.maxDifference('aaaaabbc') == 3
"""


def _lcb_spec() -> str:
    row = _LCB_ROW
    return (
        row["question_content"]
        + "\n\nComplete this starter code:\n"
        + row["starter_code"]
    )


def _state_with_public_checks() -> dict[str, Any]:
    public = build_public_assert_checks(_LCB_ROW)
    return {
        "task": _lcb_spec(),
        "entry_point": "maxDifference",
        "public_checks": public,
        "subtasks": [],
    }


def _run_sandbox_probe(name: str, code: str, state: dict[str, Any]) -> Feedback:
    probe, _fired, _resolved = build_code_probe(name, code, state)
    fb = run_in_sandbox(probe)
    return apply_oracle_fail_closed(_fired, _resolved, fb)


class _FakeEngine:
    def __init__(self, integrated_code: str) -> None:
        self._code = integrated_code

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {"integrated_code": self._code, "code_results": {}}


def test_smoke_bench_scoring_bare_def_passes() -> None:
    public = build_public_assert_checks(_LCB_ROW)
    task = BenchTask(
        task_id="3753",
        description=_lcb_spec(),
        test_code=public,
        public_checks=public,
        entry_point="maxDifference",
    )
    result = asyncio.run(
        run_benchmark([task], _FakeEngine(CORRECT_MAXDIFF), {"run_config": {"max_phase_iterations": 3}})
    )
    assert result.passed_tasks == 1


def test_smoke_bench_scoring_rejects_solution_only() -> None:
    public = build_public_assert_checks(_LCB_ROW)
    class_code = (
        "class Solution:\n"
        "    def maxDifference(self, s: str) -> int:\n"
        "        return 0\n"
    )
    full = strip_self_tests(class_code) + "\n\n" + public
    fb = run_in_sandbox(full)
    assert fb.exit_code != 0


def test_smoke_legacy_solution_test_code_fails_bare_def() -> None:
    """Documents pre-RC-1 bug: Solution()-style test_code rejects correct bare def."""
    full = strip_self_tests(CORRECT_MAXDIFF) + "\n\n" + _LEGACY_SOLUTION_TEST
    fb = run_in_sandbox(full)
    assert fb.exit_code != 0
    assert "Solution" in fb.stderr


def test_smoke_integrate_probe_fires_on_lcb_spec() -> None:
    state = _state_with_public_checks()
    _probe, fired, resolved = build_code_probe("", CORRECT_MAXDIFF, state)
    assert resolved is True
    assert fired is True
    _, legacy_fired = build_probe(CORRECT_MAXDIFF, state["task"], "maxDifference")
    assert legacy_fired is False


def test_smoke_wrong_logic_fails_integrate_probe() -> None:
    state = _state_with_public_checks()
    fb = _run_sandbox_probe("", WRONG_MAXMIN, state)
    assert fb.exit_code != 0
    assert "AssertionError" in fb.stderr or "assert" in fb.stderr.lower()


def test_smoke_correct_logic_passes_integrate_probe() -> None:
    state = _state_with_public_checks()
    fb = _run_sandbox_probe("", CORRECT_MAXDIFF, state)
    assert fb.exit_code == 0


def test_smoke_public_checks_override_wrong_model_acceptance() -> None:
    state = _state_with_public_checks()
    resolved = resolve_in_loop_check("maxDifference", MODEL_ACCEPTANCE, state)
    assert resolved == state["public_checks"]
    good_fb = _run_sandbox_probe("maxDifference", CORRECT_MAXDIFF, state)
    bad_fb = _run_sandbox_probe("maxDifference", WRONG_MAXMIN, state)
    assert good_fb.exit_code == 0
    assert bad_fb.exit_code != 0


def test_smoke_fail_closed_when_checks_unrunnable() -> None:
    fb = Feedback(stdout="", stderr="", exit_code=0)
    out = apply_oracle_fail_closed(False, True, fb)
    assert out.exit_code == 1
    assert "oracle" in out.stderr.lower()


def test_smoke_resolve_shipped_rejects_helper_blob() -> None:
    task = BenchTask(
        task_id="3777",
        description="x",
        test_code="assert maxDifference('a') == 0",
        entry_point="maxDifference",
    )
    final_state = {
        "integrated_code": "",
        "best_code": {
            "helper": "def count_frequencies(s):\n    return {}\n",
        },
    }
    assert resolve_shipped_code(final_state, task) == ""

    result = asyncio.run(
        run_benchmark(
            [task],
            _FakeEngine(""),
            {"run_config": {"max_phase_iterations": 3}},
        )
    )
    assert result.passed_tasks == 0
    assert "entry_point" in result.per_task[0].stderr


def test_smoke_mbpp_doctest_path_unchanged_when_public_checks_empty() -> None:
    mbpp_path = Path("benchmarks/mbpp160_tasks.json")
    row = json.loads(mbpp_path.read_text())[0]
    checks = extract_public_checks(row["description"], row["entry_point"])
    assert checks != ""
    resolved = resolve_in_loop_check(
        row["entry_point"],
        "",
        {
            "entry_point": row["entry_point"],
            "public_checks": "",
            "task": row["description"],
        },
    )
    assert resolved == ""
