"""In-loop signal fixes that let the engine BEAT base on doctest-bearing tasks
without regressing (issue #52 HumanEval+ follow-up):

A. extract_public_checks must not swallow a docstring's closing triple-quote into
   the doctest expected-value (the fib4 SyntaxError).
B. run_benchmark must derive public_checks from the spec doctests when the task
   ships none (so the in-loop oracle + ship gate have a trustworthy signal).
C. the graded ENTRY point is gated only by trusted public examples; a bogus
   model-authored acceptance_check must never become its correctness oracle
   (helper subtasks may still use theirs).
"""

from __future__ import annotations

import asyncio
from typing import Any

from rune.bench.runner import BenchTask, run_benchmark
from rune.engine.graph import build_code_probe, resolve_in_loop_check
from rune.engine.oracle import extract_public_checks
from rune.engine.state import Subtask, make_initial_state
from rune.sandbox.executor import run_in_sandbox

# A HumanEval-style prompt: a function whose docstring ends with a doctest
# immediately followed by the closing triple-quote.
_FIB4_PROMPT = '''def fib4(n: int):
    """The Fib4 number sequence.
    >>> fib4(5)
    4
    >>> fib4(6)
    8
    >>> fib4(7)
    14
    """
'''

_FIB4_OK = (
    "def fib4(n):\n"
    "    a, b, c, d = 0, 0, 2, 0\n"
    "    if n < 4:\n"
    "        return [0, 0, 2, 0][n]\n"
    "    for _ in range(4, n + 1):\n"
    "        a, b, c, d = b, c, d, a + b + c + d\n"
    "    return d\n"
)


def test_A_extract_public_checks_no_trailing_quote_in_want() -> None:
    checks = extract_public_checks(_FIB4_PROMPT, "fib4")
    assert checks, "expected doctest checks to be extracted"
    assert '"""' not in checks and "'''" not in checks, checks
    # The extracted checks must be runnable against a correct impl.
    probe = _FIB4_OK + "\n\n" + checks
    assert run_in_sandbox(probe, timeout=10).exit_code == 0


class _FakeEngine:
    """Captures the public_checks the engine actually receives in its state."""

    def __init__(self) -> None:
        self.seen_public = "<unset>"

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.seen_public = state["public_checks"]
        return {"integrated_code": _FIB4_OK, "zeroshot_code": {"fib4": _FIB4_OK}}


def test_B_run_benchmark_derives_public_checks_from_spec_doctests() -> None:
    task = BenchTask(
        task_id="fib4",
        description=_FIB4_PROMPT,
        test_code="assert fib4(7) == 14\n",
        entry_point="fib4",
        public_checks="",  # task ships none; engine must derive from spec
    )
    eng = _FakeEngine()
    asyncio.run(
        run_benchmark(
            [task],
            eng,
            {
                "run_config": {
                    "max_phase_iterations": 3,
                    "merge_spec_public_checks": True,
                }
            },
        )
    )
    assert eng.seen_public.strip(), "engine should have received derived public checks"
    assert "fib4" in eng.seen_public


def test_C_entry_point_ignores_untrusted_model_acceptance_check() -> None:
    # No wired/derived public signal; the decompose model attached a bogus
    # acceptance_check (undefined `debug`). It must NOT become the entry's
    # correctness oracle — the entry has no in-loop gate (module-load only), so a
    # correct, module-loadable zero-shot is accepted rather than wrongly rejected.
    state = make_initial_state("spec without doctests", 12, "f", "", "")
    state["subtasks"] = [
        Subtask(
            name="f",
            description="",
            depends_on=[],
            acceptance_check="debug(f(0))",
            builds="f",
        )
    ]
    assert resolve_in_loop_check("f", "debug(f(0))", state) == ""
    code = "def f(x):\n    return x + 1\n"
    probe, _fired, resolved = build_code_probe("f", code, state)
    assert resolved is False
    assert run_in_sandbox(probe, timeout=10).exit_code == 0


def test_C_helper_subtask_still_uses_its_acceptance_check() -> None:
    # A non-entry helper has no public examples of its own, so its model-authored
    # acceptance_check remains its only in-loop signal.
    state = make_initial_state("spec", 12, "entry", "", "")
    assert (
        resolve_in_loop_check("helper", "assert helper(1) == 2", state)
        == "assert helper(1) == 2"
    )
