import pytest
from shared.template_loader import render_prompt, render_trajectory

PYTHON_SPECIFIC_STRINGS = [
    "unittest.main()",
    "unittest.TestCase",
    "import unittest",
    "if __name__",
    "pytest",
    ".py",
]

@pytest.mark.parametrize("phase", ["code", "code_continue", "code_repair", "code_retry", "integrate", "integrate_retry"])
def test_prompt_templates_language_agnostic(phase: str) -> None:
    """Prompt templates must not contain language-specific directives."""
    kwargs: dict[str, object] = {
        "subtask_name": "test-subtask",
        "project_label": "test-project",
        "subtask_count": 3,
        "passed": 2,
        "total": 3,
        "fix_guidance": "fix the bug",
        "diagnosis": "null pointer",
        "existing_code_tail": "...",
    }
    rendered = render_prompt(phase, **kwargs)
    for needle in PYTHON_SPECIFIC_STRINGS:
        assert needle.lower() not in rendered.lower(), (
            f"prompt_{phase}.j2 contains Python-specific '{needle}'"
        )


def test_decompose_trajectory_allows_single_subtask() -> None:
    """Decompose trajectory template should mention 1-6, not 3-6."""
    rendered = render_trajectory(
        "decompose", project="Build a fibonacci function"
    )
    assert "1-6" in rendered or "1 to 6" in rendered
    assert "3-6" not in rendered
