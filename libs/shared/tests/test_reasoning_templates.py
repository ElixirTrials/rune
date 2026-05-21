"""Tests for reasoning loop Jinja2 templates."""

from shared.template_loader import render_prompt, render_trajectory


def test_artifact_compress_renders():
    text = render_trajectory(
        "artifact_compress",
        import_block="import os",
        interface_summary="def main()",
        patches="Turn 0: initial (+main)",
        test_results="1/1 passed",
        stderr_summary="",
        code_skeleton="def main(): ...",
    )
    assert "import os" in text
    assert "def main()" in text


def test_trajectory_compress_renders():
    text = render_trajectory(
        "trajectory_compress",
        turn=3,
        output="plan text here",
        feedback="",
        diagnosis="good plan",
    )
    assert "plan text here" in text


def test_prompt_reasoning_continue_renders():
    text = render_prompt(
        "reasoning_continue",
        task_description="Build a stats library",
        current_phase="code",
        turn=2,
    )
    assert "stats library" in text or "continue" in text.lower()
