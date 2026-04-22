"""Test --output-json flag on scripts/rune_runner.py.

Verifies that when ``--output-json PATH`` is set, the return dict of
``run_phased_pipeline`` is serialized to PATH as JSON with phase keys
accessible under the documented schema.

Consumed by: docs/superpowers/plans/2026-04-22-phase-corpus-producer.md
(Plan C), which shells out to rune_runner and parses the JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _fake_pipeline_result() -> dict[str, object]:
    """Minimal run_phased_pipeline return dict shape.

    Keys mirror scripts/rune_runner.py::run_phased_pipeline return statement.
    """
    return {
        "session_id": "rune-test1234",
        "total_iterations": 3,
        "project_prompt": "2+2",
        "final_tests_passed": True,
        "phases": {
            "decompose": {
                "subtasks": [{"name": "s1", "description": "d"}],
                "adapter_id": "a-decompose",
                "iterations": 1,
                "best_score": 1.0,
            },
            "plan": {
                "plans": {"s1": "plan text"},
                "plan_lengths": {"s1": 9},
                "iterations": 1,
                "best_score": 1.0,
            },
            "code": {
                "outputs": {"s1": "def f():\n    return 4\n"},
                "subtask_results": [{"name": "s1", "passed": True}],
                "iterations": 1,
                "passed": 1,
                "total": 1,
            },
            "integrate": {
                "adapter_id": "a-integrate",
                "tests_passed": True,
                "iterations": 1,
                "best_score": 1.0,
            },
        },
        "adapter_dir": "/tmp/rune-test/adapters",
        "subtasks": ["s1"],
        "adapters": [],
        "accumulated_code": "def f():\n    return 4\n",
        "evolution": {"phase_iterations": {}, "sweeps": {}, "best_adapters": {}},
    }


@pytest.fixture()
def stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Stub run_phased_pipeline to avoid loading a real model."""
    import rune_runner  # type: ignore[import-not-found]

    result = _fake_pipeline_result()

    async def _fake(**_: object) -> dict[str, object]:
        return result

    monkeypatch.setattr(rune_runner, "run_phased_pipeline", _fake)
    return result


def test_output_json_writes_file_with_phase_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline: dict[str, object],
) -> None:
    """--output-json PATH writes the pipeline result dict to PATH as JSON."""
    import rune_runner  # type: ignore[import-not-found]

    out_path = tmp_path / "pipeline_output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rune_runner.py",
            "--project",
            "2+2",
            "--output-json",
            str(out_path),
            "--max-phase-iterations",
            "1",
        ],
    )

    rune_runner.main()

    assert out_path.exists(), "output JSON file was not written"
    data = json.loads(out_path.read_text())

    phases = data.get("phases")
    assert isinstance(phases, dict), "top-level 'phases' must be a dict"
    for phase in ("decompose", "plan", "code", "integrate"):
        assert phase in phases, f"phase '{phase}' missing from output JSON"

    assert data["project_prompt"] == "2+2"
    assert data["session_id"] == "rune-test1234"


def test_no_output_json_writes_no_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline: dict[str, object],
) -> None:
    """Without --output-json, no JSON file is written and main() returns cleanly."""
    import rune_runner  # type: ignore[import-not-found]

    out_path = tmp_path / "should_not_exist.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rune_runner.py",
            "--project",
            "2+2",
            "--max-phase-iterations",
            "1",
        ],
    )

    rune_runner.main()

    assert not out_path.exists(), "no file should be written without --output-json"
