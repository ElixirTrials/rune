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

_LCB_JSONL = Path("/tmp/lcb/test6.jsonl")

pytestmark = pytest.mark.skipif(
    not _LCB_JSONL.exists(),
    reason="requires /tmp/lcb/test6.jsonl + /tmp/goal3 session data (not in CI)",
)


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


def _overnight_decompose_raw(qid: str) -> str:
    p = Path(f"/tmp/goal3/overnight/lcb_escalate_sessions/{qid}/session.jsonl")
    return json.loads(p.read_text().splitlines()[0])["output"]


def _overnight_step_code(qid: str, step: int, target: str | None = None) -> str:
    for line in (
        Path(f"/tmp/goal3/overnight/lcb_escalate_sessions/{qid}/session.jsonl")
        .read_text()
        .splitlines()
    ):
        o = json.loads(line)
        if (
            o.get("step") == step
            and (target is None or o.get("target") == target)
            and o.get("action") in ("code", "repair", "integrate")
        ):
            return o["output"]
    raise KeyError((qid, step, target))


@pytest.mark.parametrize("qid", ["3753", "3754", "3777"])
def test_decompose_collapses_to_entry_point(qid: str) -> None:
    """Overnight multi-helper decompose must collapse to one entry subtask."""
    _, fn, task = _lcb_task(qid)
    raw = _overnight_decompose_raw(qid)
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


def test_q3753_replay_ships_step4_not_integrate() -> None:
    """Step-4 repair passed LCB public; fixes must ship it not integrate regression."""
    _, fn, task = _lcb_task("3753")
    step4 = _overnight_step_code("3753", 4, "maxDifference")
    integrate = _overnight_step_code("3753", 9)

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
            "best_code": {fn: step4},
            "best_quality": {fn: 3},
            "integrated_code": integrate,
        }
    )
    shipped = resolve_shipped_code(state, task)
    assert shipped.strip() == step4.strip()
    full = shipped + "\n\n" + task.test_code
    assert run_in_sandbox(full, timeout=10).exit_code == 0


def test_q3754_replay_subtask_name_is_maxdistance() -> None:
    raw = _overnight_decompose_raw("3754")
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


def test_validity_rejects_q3754_grid_integrate() -> None:
    _, fn, task = _lcb_task("3754")
    grid = _overnight_step_code("3754", 9)
    vr = validate_solution(
        grid,
        entry_point=fn,
        signature=task.signature,
        spec=task.description,
        public_checks=task.public_checks,
    )
    assert not vr.ok
    assert any("signature" in d or "contract" in d for d in vr.deficiencies)


def test_integrate_oracle_fires_with_public_checks() -> None:
    _, fn, task = _lcb_task("3754")
    grid_integrate = _overnight_step_code("3754", 9)
    state = make_initial_state(
        task.description, 12, fn, task.signature, task.public_checks
    )
    probe, fired, resolved = build_code_probe("", grid_integrate, state)
    assert fired and resolved
    assert run_in_sandbox(probe, timeout=10).exit_code != 0


@pytest.mark.parametrize("qid", ["3748", "3799", "3801"])
def test_single_subtask_controls_still_one_subtask(qid: str) -> None:
    """Problems that already decomposed to one subtask stay stable."""
    raw = _overnight_decompose_raw(qid)
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
