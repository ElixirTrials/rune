"""Prompt and trajectory template variants for optimization.

Each variant is a function that takes an EvalTask and returns the
rendered string. This keeps templates as pure functions, testable
without the Jinja2 loader.
"""

from __future__ import annotations

from task_pool import EvalTask

# ---------------------------------------------------------------------------
# Prompt templates — how we instruct the model
# ---------------------------------------------------------------------------

_PROJECT_LABEL_CACHE: dict[str, str] = {}


def _project_label(task: EvalTask) -> str:
    """First sentence of project as label."""
    if task.name not in _PROJECT_LABEL_CACHE:
        dot = task.project.find(".")
        _PROJECT_LABEL_CACHE[task.name] = (
            task.project[: dot + 1] if dot > 0 else task.project.split("\n")[0]
        )
    return _PROJECT_LABEL_CACHE[task.name]


def prompt_minimal(task: EvalTask) -> str:
    """Minimal directive — lets adapter steer."""
    return (
        f"You are implementing: {task.subtask}\n"
        f"Project: {_project_label(task)}\n"
        "Write complete Python code with tests.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )


def prompt_must_include(task: EvalTask) -> str:
    """Lists required structural elements."""
    return (
        f"You are implementing: {task.subtask}\n"
        f"Project: {_project_label(task)}\n"
        "Follow the architecture plan in your context.\n\n"
        "Your code MUST include:\n"
        "- import unittest\n"
        "- class with methods matching the plan\n"
        "- class Test*(unittest.TestCase) with 3+ tests\n"
        "- if __name__ == '__main__': unittest.main()\n"
    )


def prompt_skeleton(task: EvalTask) -> str:
    """Provides code skeleton to complete."""
    return (
        f"You are implementing: {task.subtask}\n"
        f"Project: {_project_label(task)}\n\n"
        "Complete this skeleton:\n\n"
        "import unittest\n\n"
        f"class Test{task.subtask.replace(' ', '')}(unittest.TestCase):\n"
        "    def test_basic(self):\n"
        "        # implement\n\n"
        f"class {task.subtask.replace(' ', '')}:\n"
        "    def __init__(self):\n"
        "        # implement\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def prompt_hybrid(task: EvalTask) -> str:
    """Domain context in prompt + structural requirements."""
    return (
        f"You are implementing: {task.subtask}\n"
        f"Project: {_project_label(task)}\n"
        f"Plan: {task.plan}\n\n"
        "Write self-contained Python with unittest.TestCase tests.\n"
        "import unittest at top. End with unittest.main().\n"
    )


def prompt_open(task: EvalTask) -> str:
    """Open-ended — maximum adapter reliance."""
    return (
        "Write Python code for the system described in your context.\n"
        "Include unittest.TestCase tests.\n"
        "End with: if __name__ == '__main__': unittest.main()"
    )


PROMPT_STYLES = {
    "minimal": prompt_minimal,
    "must_include": prompt_must_include,
    "skeleton": prompt_skeleton,
    "hybrid": prompt_hybrid,
    "open": prompt_open,
}


def get_prompt(style: str, task: EvalTask) -> str:
    """Render a prompt by style name."""
    fn = PROMPT_STYLES.get(style)
    if fn is None:
        raise ValueError(f"Unknown prompt style: {style}. Choose from {list(PROMPT_STYLES)}")
    return fn(task)


# ---------------------------------------------------------------------------
# Trajectory templates — what gets encoded into the adapter
# ---------------------------------------------------------------------------


def trajectory_prose(task: EvalTask) -> str:
    """Natural language description."""
    return (
        f"PROJECT: {task.project}\n"
        f"SUBTASK: {task.subtask}\n"
        f"PLAN: {task.plan}\n"
    )


def trajectory_exemplar(task: EvalTask) -> str:
    """Code-like exemplar showing expected patterns."""
    return (
        f"# {task.project}\n"
        f"# Subtask: {task.subtask}\n"
        "import unittest\n"
        "from dataclasses import dataclass\n\n"
        f"class {task.subtask.replace(' ', '')}:\n"
        f"    '''Implementation of {task.subtask}.'''\n"
        "    def __init__(self): ...\n\n"
        f"class Test{task.subtask.replace(' ', '')}(unittest.TestCase):\n"
        "    def test_basic(self): ...\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def trajectory_signatures(task: EvalTask) -> str:
    """Class/function signatures only — compact."""
    return (
        f"PROJECT: {task.project}\n"
        f"SUBTASK: {task.subtask}\n"
        f"INTERFACES:\n"
        f"  class {task.subtask.replace(' ', '')}:\n"
        f"    def __init__(self): ...\n"
        f"TESTS:\n"
        f"  class Test{task.subtask.replace(' ', '')}(unittest.TestCase):\n"
        f"    def test_basic(self): ...\n"
    )


def trajectory_full_context(task: EvalTask) -> str:
    """Maximum context — project + plan + practices."""
    return (
        f"PROJECT: {task.project}\n\n"
        f"SUBTASK: {task.subtask}\n"
        f"PLAN: {task.plan}\n\n"
        "PRACTICES:\n"
        "- Use dataclasses for models, type annotations\n"
        "- unittest.TestCase for tests, test-first pattern\n"
        "- Self-contained file, no external imports\n"
        "- import unittest at top\n"
        "- End with if __name__ == '__main__': unittest.main()\n"
    )


def trajectory_minimal(task: EvalTask) -> str:
    """Just names — minimal adapter signal."""
    return f"{task.project[:80]}\n{task.subtask}\n"


TRAJECTORY_STYLES = {
    "prose": trajectory_prose,
    "exemplar": trajectory_exemplar,
    "signatures": trajectory_signatures,
    "full_context": trajectory_full_context,
    "minimal": trajectory_minimal,
}


def get_trajectory(style: str, task: EvalTask) -> str:
    """Render a trajectory by style name."""
    fn = TRAJECTORY_STYLES.get(style)
    if fn is None:
        raise ValueError(
            f"Unknown trajectory style: {style}. Choose from {list(TRAJECTORY_STYLES)}"
        )
    return fn(task)
