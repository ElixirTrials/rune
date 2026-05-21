"""Tests for template truncation limits.

Verifies that trajectory templates use appropriate truncation limits
so the hypernetwork perceiver receives adequate context within
the ~8000 char adapter capacity.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "src" / "shared" / "templates"


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))


def test_code_template_plan_limit() -> None:
    """code.j2 should include at least 1200 chars of plan text."""
    env = _env()
    tmpl = env.get_template("code.j2")
    long_plan = "P" * 1300
    rendered = tmpl.render(
        subtask={"name": "test"},
        subtask_index=1,
        total_subtasks=1,
        plan=long_plan,
        existing_code="",
    )
    assert "P" * 1200 in rendered
    assert "P" * 1201 not in rendered


def test_code_template_existing_code_limit() -> None:
    """code.j2 should include at least 2000 chars of existing_code."""
    env = _env()
    tmpl = env.get_template("code.j2")
    long_code = "C" * 2100
    rendered = tmpl.render(
        subtask={"name": "test"},
        subtask_index=1,
        total_subtasks=1,
        plan="short plan",
        existing_code=long_code,
    )
    assert "C" * 2000 in rendered
    assert "C" * 2001 not in rendered


def test_code_retry_template_plan_limit() -> None:
    """code_retry.j2 should include at least 1000 chars of plan text."""
    env = _env()
    tmpl = env.get_template("code_retry.j2")
    long_plan = "P" * 1100
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
    assert "P" * 1000 in rendered
    assert "P" * 1001 not in rendered


def test_code_retry_template_error_limit() -> None:
    """code_retry.j2 should include at least 500 chars of error_summary."""
    env = _env()
    tmpl = env.get_template("code_retry.j2")
    long_error = "E" * 600
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
    assert "E" * 500 in rendered
    assert "E" * 501 not in rendered


def test_decompose_template_project_limit() -> None:
    """decompose.j2 should include at least 1500 chars of project text."""
    env = _env()
    tmpl = env.get_template("decompose.j2")
    long_project = "D" * 1600
    rendered = tmpl.render(project=long_project)
    assert "D" * 1500 in rendered
    assert "D" * 1501 not in rendered


def test_plan_template_description_limit() -> None:
    """plan.j2 should include at least 600 chars of description."""
    env = _env()
    tmpl = env.get_template("plan.j2")
    long_desc = "X" * 700
    rendered = tmpl.render(
        subtask={"name": "test", "description": long_desc},
        subtask_index=1,
        total_subtasks=1,
        project="short project",
    )
    assert "X" * 600 in rendered
    assert "X" * 601 not in rendered


def test_integrate_template_skeleton_limit() -> None:
    """integrate.j2 should include at least 1200 chars of skeleton text."""
    env = _env()
    tmpl = env.get_template("integrate.j2")
    long_skeleton = "S" * 1300
    rendered = tmpl.render(
        project="test project",
        subtask_count=1,
        skeletons={"task1": long_skeleton},
    )
    assert "S" * 1200 in rendered
    assert "S" * 1201 not in rendered


def test_integrate_template_project_limit() -> None:
    """integrate.j2 should include at least 400 chars of project text."""
    env = _env()
    tmpl = env.get_template("integrate.j2")
    long_project = "P" * 500
    rendered = tmpl.render(
        project=long_project,
        subtask_count=1,
        skeletons={"task1": "skeleton"},
    )
    assert "P" * 400 in rendered
    assert "P" * 401 not in rendered


def test_code_continue_existing_code_limit() -> None:
    """code_continue.j2 should include at least 5000 chars of existing_code."""
    env = _env()
    tmpl = env.get_template("code_continue.j2")
    long_code = "C" * 5100
    rendered = tmpl.render(
        subtask={"name": "test"},
        attempt=1,
        max_retries=1,
        existing_code=long_code,
    )
    assert "C" * 5000 in rendered
    assert "C" * 5001 not in rendered


def test_code_repair_existing_code_limit() -> None:
    """code_repair.j2 should include at least 2000 chars of existing_code."""
    env = _env()
    tmpl = env.get_template("code_repair.j2")
    long_code = "C" * 2100
    rendered = tmpl.render(
        subtask={"name": "test"},
        diagnosis="fix the bug",
        existing_code=long_code,
        sibling_skeletons={},
    )
    assert "C" * 2000 in rendered
    assert "C" * 2001 not in rendered
