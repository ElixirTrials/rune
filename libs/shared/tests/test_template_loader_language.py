from pathlib import Path

from shared.template_loader import render_trajectory


def test_render_trajectory_default_language() -> None:
    """render_trajectory without language param returns base template."""
    result = render_trajectory("code", project="test", subtask={"name": "test", "description": "test"}, subtask_index=1, total_subtasks=1, plan="plan")
    assert "ROLE: coder" in result


def test_render_trajectory_unknown_language_falls_back() -> None:
    """render_trajectory with unknown language falls back to base template."""
    result = render_trajectory("code", language="rust", project="test", subtask={"name": "test", "description": "test"}, subtask_index=1, total_subtasks=1, plan="plan")
    assert "ROLE: coder" in result


def test_render_trajectory_picks_language_template() -> None:
    """When a language-specific template exists, it is preferred."""
    templates_dir = Path(__file__).resolve().parents[1] / "src" / "shared" / "templates"
    lang_template = templates_dir / "code.testlang.j2"
    lang_template.write_text("ROLE: coder-testlang\nLANG: testlang\n")
    try:
        result = render_trajectory(
            "code", language="testlang",
            project="test", subtask={"name": "t", "description": "t"},
            subtask_index=1, total_subtasks=1, plan="p",
        )
        assert "coder-testlang" in result
    finally:
        lang_template.unlink()
