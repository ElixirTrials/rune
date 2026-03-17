"""Tests for template truncation limits.

Verifies that trajectory templates use the increased truncation limits
so the hypernetwork perceiver receives adequate context.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "src" / "shared" / "templates"


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))


def test_code_template_plan_limit() -> None:
    """code.j2 should include at least 500 chars of plan text."""
    env = _env()
    tmpl = env.get_template("code.j2")
    long_plan = "P" * 600
    rendered = tmpl.render(
        subtask={"name": "test"},
        subtask_index=1,
        total_subtasks=1,
        plan=long_plan,
        existing_code="",
    )
    assert "P" * 500 in rendered
    assert "P" * 501 not in rendered


def test_code_template_existing_code_limit() -> None:
    """code.j2 should include at least 400 chars of existing_code."""
    env = _env()
    tmpl = env.get_template("code.j2")
    long_code = "C" * 500
    rendered = tmpl.render(
        subtask={"name": "test"},
        subtask_index=1,
        total_subtasks=1,
        plan="short plan",
        existing_code=long_code,
    )
    assert "C" * 400 in rendered
    assert "C" * 401 not in rendered


def test_code_retry_template_plan_limit() -> None:
    """code_retry.j2 should include at least 400 chars of plan text."""
    env = _env()
    tmpl = env.get_template("code_retry.j2")
    long_plan = "P" * 500
    rendered = tmpl.render(
        subtask={"name": "test"},
        attempt=1,
        max_retries=3,
        plan=long_plan,
        existing_code="",
        passed=0,
        total=1,
        tests_passed=False,
        error_summary="err",
        failed_tests="",
        fix_guidance="fix it",
        history="",
    )
    assert "P" * 400 in rendered
    assert "P" * 401 not in rendered


def test_code_retry_template_error_limit() -> None:
    """code_retry.j2 should include at least 300 chars of error_summary."""
    env = _env()
    tmpl = env.get_template("code_retry.j2")
    long_error = "E" * 400
    rendered = tmpl.render(
        subtask={"name": "test"},
        attempt=1,
        max_retries=3,
        plan="plan",
        existing_code="",
        passed=0,
        total=1,
        tests_passed=False,
        error_summary=long_error,
        failed_tests="",
        fix_guidance="fix it",
        history="",
    )
    assert "E" * 300 in rendered
    assert "E" * 301 not in rendered


def test_decompose_template_project_limit() -> None:
    """decompose.j2 should include at least 1200 chars of project text."""
    env = _env()
    tmpl = env.get_template("decompose.j2")
    long_project = "D" * 1400
    rendered = tmpl.render(project=long_project)
    assert "D" * 1200 in rendered
    assert "D" * 1201 not in rendered


def test_plan_template_description_limit() -> None:
    """plan.j2 should include at least 500 chars of description."""
    env = _env()
    tmpl = env.get_template("plan.j2")
    long_desc = "X" * 600
    rendered = tmpl.render(
        subtask={"name": "test", "description": long_desc},
        subtask_index=1,
        total_subtasks=1,
        project="short project",
    )
    assert "X" * 500 in rendered
    assert "X" * 501 not in rendered
