import json
from pathlib import Path

from rune.engine.state import Feedback, StepRecord
from rune.mining.session_log import write_session


def _state() -> dict:
    return {
        "trajectory": [
            StepRecord(
                step=0,
                action_name="code",
                target_subtask="_main",
                adapter_id="a0",
                feedback=Feedback(stdout="", stderr="", exit_code=0),
                generated_code="print(1)",
                trajectory_text="ROLE: coder",
                prompt_text="write a printer",
                output_text="print(1)",
            )
        ]
    }


def test_write_session_emits_jsonl_and_metadata(tmp_path: Path) -> None:
    out = write_session(
        _state(),
        {"benchmark": "mbpp", "problem_id": "7"},
        tmp_path / "sess",
    )
    lines = (out / "session.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "code"
    assert rec["target"] == "_main"
    assert rec["trajectory"] == "ROLE: coder"
    assert rec["prompt"] == "write a printer"
    assert rec["output"] == "print(1)"
    assert rec["feedback"]["exit_code"] == 0
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["benchmark"] == "mbpp"
    assert meta["problem_id"] == "7"
    assert meta["schema_version"] == 2  # stamped by write_session, not the caller
