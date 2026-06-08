"""RFT corpus mining: successful repair/solve episodes -> distill corpus rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_rft_mine", Path(__file__).resolve().parents[2] / "tools" / "_rft_mine.py"
)
_rft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rft)


def _write_session(d: Path, steps: list[dict], pass_at_1: bool = True) -> None:
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(
        json.dumps({"problem_id": "mbpp/1", "pass_at_1": pass_at_1})
    )
    (d / "session.jsonl").write_text("\n".join(json.dumps(s) for s in steps))


def _step(action, exit_code, code, target="f"):
    return {
        "action": action,
        "target": target,
        "trajectory": "## Mission `f`\nsolve it\n## `f` — what you learned was wrong\nboom",
        "output": f"```python\n{code}\n```",
        "feedback": {"exit_code": exit_code, "stdout": "", "stderr": ""},
    }


class TestMine:
    def test_repair_that_fixes_is_mined(self, tmp_path: Path) -> None:
        d = tmp_path / "s"
        _write_session(
            d,
            [
                _step("code", 1, "def f(x): return 0"),  # fails
                _step("diagnose", None, ""),
                _step("repair", 0, "def f(x): return x"),  # repair fixes
            ],
        )
        rows = _rft._mine_session(d, repairs_only=False, held_out_only=False)
        assert len(rows) == 1
        r = rows[0]
        assert r["entry_point"] == "f"  # derived from the code's def
        assert r["answer"] == "def f(x): return x"
        assert "## Mission" in r["context"]  # the new-format conditioning
        assert r["source_action"] == "repair"

    def test_repairs_only_excludes_first_pass_code(self, tmp_path: Path) -> None:
        d = tmp_path / "s"
        _write_session(
            d,
            [_step("code", 0, "def f(x): return x")],  # passed on first try (no repair)
        )
        assert _rft._mine_session(d, repairs_only=True, held_out_only=False) == []
        # but mined when repairs_only is off
        assert len(_rft._mine_session(d, repairs_only=False, held_out_only=False)) == 1

    def test_held_out_only_filters_failed_tasks(self, tmp_path: Path) -> None:
        d = tmp_path / "s"
        _write_session(d, [_step("repair", 0, "def f(x): return x")], pass_at_1=False)
        assert _rft._mine_session(d, repairs_only=False, held_out_only=True) == []

    def test_failing_steps_not_mined(self, tmp_path: Path) -> None:
        d = tmp_path / "s"
        _write_session(d, [_step("code", 1, "def f(x): return 0")])
        assert _rft._mine_session(d, repairs_only=False, held_out_only=False) == []
