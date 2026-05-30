from rune.engine.state import StepRecord


def test_step_record_carries_renders() -> None:
    rec = StepRecord(
        step=0,
        action_name="code",
        target_subtask="_main",
        adapter_id="a",
        feedback=None,
        generated_code="print(1)",
        trajectory_text="ROLE: coder",
        prompt_text="write code",
        output_text="print(1)",
    )
    assert rec.trajectory_text == "ROLE: coder"
    assert rec.prompt_text == "write code"
    assert rec.output_text == "print(1)"
