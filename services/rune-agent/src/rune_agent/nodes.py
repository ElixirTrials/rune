"""Node functions for the Rune coding agent recursive loop."""

import logging
import os
import re
import subprocess
import tempfile
from typing import Any

from inference import GenerationResult, get_provider
from model_training.trajectory import record_trajectory

from .state import RuneState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a Python code generator. "
    "Output only code, no explanation."
)
DEFAULT_TIMEOUT = 30
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B"


def _build_prompt(state: RuneState) -> str:
    """Build the user prompt for the LLM based on current attempt.

    First attempt (attempt_count == 0): includes task description and test suite.
    Retry (attempt_count > 0): also includes prior code, stdout, stderr, exit code.

    Args:
        state: Current agent state.

    Returns:
        Formatted prompt string.
    """
    task = state["task_description"]
    test_suite = state["test_suite"]

    base = (
        f"Task: {task}\n\n"
        f"Test suite (your code must pass these):\n{test_suite}\n\n"
        "Write a Python solution:"
    )

    if state["attempt_count"] == 0:
        return base

    prior_code = state["generated_code"]
    stdout = state["stdout"]
    stderr = state["stderr"]
    exit_code = state["exit_code"]

    return (
        f"{base}\n\n"
        "Your previous attempt produced the following errors:\n"
        f"Code:\n{prior_code}\n\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}\n"
        f"exit_code: {exit_code}\n\n"
        "Please fix the issues and write a corrected solution:"
    )


def _extract_code(text: str) -> str:
    """Extract code from a markdown python block, or return stripped text.

    Args:
        text: Raw LLM response text.

    Returns:
        Extracted code string.
    """
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


async def generate_node(state: RuneState) -> dict[str, Any]:
    """Generate code for the given task using the inference layer.

    Calls the LLM (with optional LoRA adapters) to produce a code solution
    for the task description.

    Args:
        state: Current agent state with task description and context.

    Returns:
        State update dict with generated_code key.

    Example:
        >>> state = {"task_description": "Write fibonacci", "task_type": "function",
        ...          "test_suite": "assert fib(5) == 5", "adapter_ids": []}
        >>> result = await generate_node(state)
        >>> 'generated_code' in result
        True
    """
    # Read env vars inside function body so monkeypatch.setenv() works in tests
    model: str = os.environ.get("RUNE_MODEL", DEFAULT_MODEL)
    adapter_id: str | None = state["adapter_ids"][0] if state["adapter_ids"] else None

    provider = get_provider()
    user_prompt = _build_prompt(state)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    result: GenerationResult = await provider.generate(
        prompt=full_prompt,
        model=model,
        adapter_id=adapter_id,
    )

    extracted = _extract_code(result.text)
    logger.info(
        "generate_node: attempt=%d, model=%s, adapter_id=%s, tokens=%d",
        state["attempt_count"],
        result.model,
        result.adapter_id,
        result.token_count,
    )

    return {"generated_code": extracted}


async def execute_node(state: RuneState) -> dict[str, Any]:
    """Execute the generated code in a sandboxed environment.

    Runs the generated code against the test suite in an isolated subprocess
    and captures stdout, stderr, and exit code.

    Args:
        state: Current agent state with generated code and test suite.

    Returns:
        State update dict with stdout, stderr, exit_code, and tests_passed keys.

    Example:
        >>> state = {"generated_code": "def fib(n): return n", "test_suite": ""}
        >>> result = await execute_node(state)
        >>> 'tests_passed' in result
        True
    """
    # Read env var inside function body so monkeypatch.setenv() works in tests
    timeout: int = int(os.environ.get("RUNE_EXEC_TIMEOUT", DEFAULT_TIMEOUT))

    script = state["generated_code"] + "\n\n" + state["test_suite"]

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = tmpdir + "/solution.py"
        with open(script_path, "w") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode
            tests_passed = proc.returncode == 0
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Execution timed out after {timeout}s"
            exit_code = 1
            tests_passed = False

    logger.info(
        "execute_node: exit_code=%d, tests_passed=%s",
        exit_code,
        tests_passed,
    )

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "tests_passed": tests_passed,
    }


async def reflect_node(state: RuneState) -> dict[str, Any]:
    """Reflect on execution results and record the attempt in trajectory.

    Increments the attempt counter and appends the current attempt's data
    to the trajectory list. Does not make any LLM calls.

    Args:
        state: Current agent state with execution results.

    Returns:
        State update dict with incremented attempt_count and extended trajectory.

    Example:
        >>> state = {"attempt_count": 0, "generated_code": "def fib(n): pass",
        ...          "exit_code": 1, "tests_passed": False, "trajectory": [],
        ...          "stdout": "", "stderr": ""}
        >>> result = await reflect_node(state)
        >>> result['attempt_count']
        1
    """
    step: dict[str, Any] = {
        "generated_code": state["generated_code"],
        "stdout": state["stdout"],
        "stderr": state["stderr"],
        "exit_code": state["exit_code"],
        "tests_passed": state["tests_passed"],
    }

    # Use list concatenation (not .append()) — LangGraph requires immutable state updates
    new_trajectory: list[dict[str, Any]] = state["trajectory"] + [step]
    new_attempt_count = state["attempt_count"] + 1

    logger.info(
        "reflect_node: attempt=%d -> %d, trajectory_length=%d",
        state["attempt_count"],
        new_attempt_count,
        len(new_trajectory),
    )

    return {
        "attempt_count": new_attempt_count,
        "trajectory": new_trajectory,
    }


async def save_trajectory_node(state: RuneState) -> dict[str, Any]:
    """Save the completed trajectory for parametric memory training.

    Persists the trajectory to disk via record_trajectory() and determines
    the final outcome based on whether tests passed.

    Args:
        state: Current agent state with complete trajectory and outcome.

    Returns:
        State update dict with outcome key ('success' or 'exhausted').

    Example:
        >>> state = {"tests_passed": True, "session_id": "abc", "trajectory": [],
        ...          "task_description": "", "task_type": "", "adapter_ids": []}
        >>> result = await save_trajectory_node(state)
        >>> result['outcome']
        'success'
    """
    outcome = "success" if state["tests_passed"] else "exhausted"

    record_trajectory(
        session_id=state["session_id"],
        steps=state["trajectory"],
        outcome=outcome,
        task_description=state["task_description"],
        task_type=state["task_type"],
        adapter_ids=state["adapter_ids"],
    )

    logger.info(
        "save_trajectory_node: session_id=%s, outcome=%s",
        state["session_id"],
        outcome,
    )

    return {"outcome": outcome}
