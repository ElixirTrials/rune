"""Tests for LCB engine fixes: deterministic subtask collapse, ship-best, integrate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rune.bench.lcb import build_public_assert_checks
from rune.bench.runner import BenchTask, resolve_shipped_code
from rune.engine.graph import build_code_probe, render_episode_adapter
from rune.engine.parse import parse_output
from rune.engine.policy import select_action
from rune.engine.state import Action, Subtask, make_initial_state
from rune.engine.validity import validate_solution
from rune.sandbox.executor import run_in_sandbox

# Hermetic fixtures vendored from the LCB-v6 escalate run (slimmed LCB rows +
# the step-0 decompose output per qid), so these engine-fix regression tests run
# in CI without the 134MB test6.jsonl or ephemeral /tmp GPU-run session data.
_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "lcb_engine_fixes"
_LCB_JSONL = _FIXTURES / "rows.jsonl"


def _lcb_row(qid: str) -> dict:
    for line in _LCB_JSONL.read_text().splitlines():
        row = json.loads(line)
        if row["question_id"] == qid:
            return row
    raise KeyError(qid)


def _lcb_task(qid: str) -> tuple[dict, str, BenchTask]:
    row = _lcb_row(qid)
    meta = json.loads(row["metadata"])
    fn = meta["func_name"]
    public = build_public_assert_checks(row)
    desc = row["question_content"]
    if row.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + row["starter_code"]
    task = BenchTask(
        task_id=qid,
        description=desc,
        test_code=public,
        entry_point=fn,
        signature=row.get("starter_code", ""),
        public_checks=public,
    )
    return row, fn, task


def _decompose_raw(qid: str) -> str:
    p = _FIXTURES / "sessions" / qid / "session.jsonl"
    return json.loads(p.read_text().splitlines()[0])["output"]


# A maxDistance integrate with the WRONG signature (1-arg ``grid`` vs the
# ``maxDistance(s, k)`` starter contract) — the over-decomposed-helper bug the
# engine fixes guard against (issue #52).
_Q3754_WRONG_SIG_GRID = (
    "def maxDistance(grid):\n"
    "    for row in grid:\n"
    "        for _ in row:\n"
    "            pass\n"
    "    return 0\n"
)


@pytest.mark.parametrize("qid", ["3753", "3754", "3777"])
def test_decompose_collapses_to_entry_point(qid: str) -> None:
    """Overnight multi-helper decompose must collapse to one entry subtask."""
    _, fn, task = _lcb_task(qid)
    raw = _decompose_raw(qid)
    state = make_initial_state(
        task.description,
        12,
        task.entry_point,
        task.signature,
        task.public_checks,
    )
    out = parse_output(
        Action(
            "decompose", "decompose", "prompt_decompose_concise", "", None, False, None
        ),
        raw,
        None,
        state,
    )
    subs = out["subtasks"]
    assert len(subs) == 1
    assert subs[0].name == fn
    assert subs[0].acceptance_check == task.public_checks


def test_q3753_ships_passing_best_over_regressing_integrate() -> None:
    """A best_code that passes the LCB public checks must be shipped, never a
    regressing integrate (issue #52 RC-C)."""
    _, fn, task = _lcb_task("3753")
    # Correct maxDifference: max odd-count freq minus min even-count freq.
    best = (
        "from collections import Counter\n"
        "def maxDifference(s):\n"
        "    c = Counter(s)\n"
        "    odd = max(v for v in c.values() if v % 2 == 1)\n"
        "    even = min(v for v in c.values() if v % 2 == 0)\n"
        "    return odd - even\n"
    )
    integrate = "def maxDifference(s):\n    return 0\n"  # regresses: fails public

    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    state.update(
        {
            "subtasks": [
                Subtask(
                    name=fn,
                    description="",
                    depends_on=[],
                    acceptance_check=task.public_checks,
                    builds=fn,
                )
            ],
            "best_code": {fn: best},
            "best_quality": {fn: 3},
            "integrated_code": integrate,
        }
    )
    shipped = resolve_shipped_code(state, task)
    assert shipped.strip() == best.strip()
    full = shipped + "\n\n" + task.test_code
    assert run_in_sandbox(full, timeout=10).exit_code == 0


def test_q3754_replay_subtask_name_is_maxdistance() -> None:
    raw = _decompose_raw("3754")
    _, fn, task = _lcb_task("3754")
    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    out = parse_output(
        Action(
            "decompose", "decompose", "prompt_decompose_concise", "", None, False, None
        ),
        raw,
        None,
        state,
    )
    assert [s.name for s in out["subtasks"]] == ["maxDistance"]


def test_q3777_ship_best_when_integrate_empty() -> None:
    """Helper-only join must not ship when entry missing from integrate."""
    _, fn, task = _lcb_task("3777")
    helper = "def max_product_with_alternating_sum(nums):\n    return 0\n"
    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    state.update(
        {
            "best_code": {"max_product_with_alternating_sum": helper},
            "integrated_code": "",
        }
    )
    assert resolve_shipped_code(state, task) == ""


def test_policy_skips_integrate_when_benchmark_exhausted() -> None:
    _, fn, task = _lcb_task("3753")
    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    state.update(
        {
            "subtasks": [
                Subtask(
                    name=fn,
                    description="",
                    depends_on=[],
                    acceptance_check=task.public_checks,
                    builds=fn,
                )
            ],
            "plans": {fn: "plan"},
            "code_results": {fn: "def x: pass"},
            "code_passed": {fn: False},
            "retries": {fn: 8},
            "best_code": {fn: "def maxDifference(s):\n    return 1\n"},
        }
    )
    assert select_action(state) == []


def test_integrate_adapter_uses_best_code() -> None:
    _, fn, task = _lcb_task("3754")
    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    state.update(
        {
            "subtasks": [
                Subtask(
                    name=fn,
                    description="d",
                    depends_on=[],
                    acceptance_check=task.public_checks,
                    builds=fn,
                )
            ],
            "overall_goal": "goal",
            "code_results": {fn: "def maxDistance(s,k):\n    return 0\n"},
            "best_code": {fn: "def maxDistance(s,k):\n    return 3\n"},
        }
    )
    traj = render_episode_adapter("integrate", "", state)
    assert "return 3" in traj
    assert "return 0" not in traj


def test_validity_rejects_wrong_signature_integrate() -> None:
    """An integrate whose signature violates the entry-point contract
    (``maxDistance(grid)`` vs the ``maxDistance(s, k)`` starter) must be rejected
    by validate_solution (issue #52)."""
    _, fn, task = _lcb_task("3754")
    vr = validate_solution(
        _Q3754_WRONG_SIG_GRID,
        entry_point=fn,
        signature=task.signature,
        spec=task.description,
        public_checks=task.public_checks,
    )
    assert not vr.ok
    assert any("signature" in d or "contract" in d for d in vr.deficiencies)


def test_integrate_oracle_fires_with_public_checks() -> None:
    _, fn, task = _lcb_task("3754")
    grid_integrate = _Q3754_WRONG_SIG_GRID
    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    probe, fired, resolved = build_code_probe("", grid_integrate, state)
    assert fired and resolved
    assert run_in_sandbox(probe, timeout=10).exit_code != 0


@pytest.mark.parametrize("qid", ["3748", "3799", "3801"])
def test_single_subtask_controls_still_one_subtask(qid: str) -> None:
    """Problems that already decomposed to one subtask stay stable."""
    raw = _decompose_raw(qid)
    row = _lcb_row(qid)
    fn = json.loads(row["metadata"])["func_name"]
    public = build_public_assert_checks(row)
    desc = row["question_content"]
    state = make_initial_state(desc, 12, fn, row.get("starter_code", ""), public)
    out = parse_output(
        Action(
            "decompose", "decompose", "prompt_decompose_concise", "", None, False, None
        ),
        raw,
        None,
        state,
    )
    assert len(out["subtasks"]) == 1
    assert out["subtasks"][0].name == fn
