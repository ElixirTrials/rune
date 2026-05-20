"""Node functions for the Rune coding agent recursive loop."""

import ast
import logging
import os
import re
from typing import Any

from inference import GenerationResult, get_provider
from model_training.trajectory import record_trajectory
from shared.sandbox import count_test_results, get_sandbox_backend

from .state import RuneState

logger = logging.getLogger(__name__)

_SYS_CODE = (
    "You are a code generator. Output ONLY valid executable code. "
    "No explanation, no commentary, no markdown fencing."
)
_PHASE_SYSTEM_PROMPTS: dict[str, str] = {
    "decompose": (
        "You are a project decomposer. Break projects into subtasks. "
        "Output ONLY a numbered list, never code."
    ),
    "decompose_concise": (
        "You are a project decomposer. Break projects into subtasks. "
        "Output ONLY a numbered list, never code."
    ),
    "plan": (
        "You are a software architect. Output ONLY architecture plans "
        "with class signatures, data flow, and test strategy. Never output code."
    ),
    "diagnose": (
        "You are a code diagnostician. Identify which subtasks have bugs. "
        "Output ONLY a numbered list, never code."
    ),
    "code": _SYS_CODE,
    "code_retry": _SYS_CODE,
    "code_continue": _SYS_CODE,
    "code_repair": _SYS_CODE,
    "integrate": _SYS_CODE,
    "integrate_retry": _SYS_CODE,
}
DEFAULT_SYSTEM_PROMPT = _SYS_CODE
DEFAULT_TIMEOUT = 30
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
_TEXT_ONLY_PHASES = frozenset(
    {
        "decompose",
        "decompose_concise",
        "plan",
        "diagnose",
    }
)


def _build_prompt(state: RuneState) -> str:
    """Build the user prompt for the LLM based on current attempt.

    When ``state["phase"]`` is set, renders the corresponding Jinja2 prompt
    template (e.g. ``prompt_code.j2``).  Retry context is intentionally
    omitted — errors flow through the adapter weights, not the prompt.

    When ``state["phase"]`` is ``None``, preserves the original behaviour
    for backward compatibility.

    Args:
        state: Current agent state.

    Returns:
        Formatted prompt string.
    """
    phase = state.get("phase")

    if phase is not None:
        from shared.template_loader import render_prompt

        ctx = state.get("prompt_context") or {}
        return render_prompt(phase, task_description=state["task_description"], **ctx)

    task = state["task_description"]
    test_suite = state["test_suite"]

    base = (
        f"Task: {task}\n\n"
        f"Test suite (your code must pass these):\n{test_suite}\n\n"
        "Write a solution:"
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


_CODE_START_RE = re.compile(r"^(import |from |def |class |@)", re.MULTILINE)


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _longest_valid_block(text: str) -> str:
    """Find the longest contiguous line range that is valid Python.

    Adapted from EvalPlus's code_extract — the de-facto standard for
    sanitizing LLM code output across major benchmarks. O(n²) worst case
    but n is bounded by max_tokens (typically <200 lines).
    """
    lines = text.split("\n")
    n = len(lines)
    best_start = 0
    best_end = 0
    best_nonblank = 0

    for i in range(n):
        if n - i <= best_nonblank:
            break
        for j in range(n, i, -1):
            if j - i <= best_nonblank:
                break
            candidate = "\n".join(lines[i:j])
            if _syntax_ok(candidate):
                nonblank = sum(1 for line in lines[i:j] if line.strip())
                if nonblank > best_nonblank:
                    best_nonblank = nonblank
                    best_start = i
                    best_end = j
                break

    if best_nonblank > 0:
        return "\n".join(lines[best_start:best_end]).strip()
    return ""


def _extract_code(text: str) -> str:
    """Extract executable Python from an LLM response.

    Fast path: fence extraction or raw-text validation. Fallback: find
    the longest contiguous valid-Python line range (EvalPlus pattern).
    Returns empty string if nothing valid — the retry loop handles that.
    """
    # 1. Try ```python fence (prefer last closed fence).
    fenced = list(re.finditer(r"```python\s*(.*?)```", text, re.DOTALL))
    if fenced:
        code = fenced[-1].group(1).strip()
        if _syntax_ok(code):
            return code

    # 2. Truncated/unclosed fence (output hit max_tokens).
    trunc = re.search(r"```python\s*(.*)", text, re.DOTALL)
    if trunc:
        code = trunc.group(1).strip()
        if _syntax_ok(code):
            return code

    # 3. Raw text as-is.
    raw = text.strip()
    if _syntax_ok(raw):
        return raw

    # 4. Fast heuristic: strip preamble at first code-like line.
    m = _CODE_START_RE.search(raw)
    if m:
        tail = raw[m.start() :].strip()
        if _syntax_ok(tail):
            return tail

    # 5. Robust fallback: longest contiguous valid block.
    result = _longest_valid_block(raw)
    if result:
        return result

    # 6. Unrecoverable — return empty to trigger retry.
    logger.warning("_extract_code: could not recover valid Python from LLM output")
    return ""


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
    phase = state.get("phase")
    system_prompt = _PHASE_SYSTEM_PROMPTS.get(phase or "", DEFAULT_SYSTEM_PROMPT)

    # Per-phase max_tokens: RUNE_MAX_TOKENS_CODE, RUNE_MAX_TOKENS_INTEGRATE, etc.
    phase_key = (phase or "").upper().replace("-", "_")
    max_tokens = int(
        os.environ.get(f"RUNE_MAX_TOKENS_{phase_key}", "")
        or os.environ.get("RUNE_MAX_TOKENS", "2048")
    )

    # Qwen3.5-9B ships with thinking disabled — the 9B was not tuned for
    # thinking as its primary mode.  Forcing enable_thinking=True causes
    # meta-reasoning about system prompts instead of following them.
    enable_thinking = False

    # Qwen3.5-9B recommended: temp=0.7, top_p=0.8 for non-thinking mode.
    # Text-only phases (decompose, plan, diagnose) use slightly higher
    # temperature for diversity; code phases keep the provider defaults.
    if phase in _TEXT_ONLY_PHASES:
        temperature: float | None = 0.7
        top_p: float | None = 0.8
    else:
        temperature = None
        top_p = None
    thinking_budget = 0

    result: GenerationResult = await provider.generate(
        prompt=user_prompt,
        model=model,
        adapter_id=adapter_id,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )

    extracted = (
        result.text.strip()
        if phase in _TEXT_ONLY_PHASES
        else _extract_code(result.text)
    )
    logger.info(
        "generate_node: attempt=%d, model=%s, adapter_id=%s, tokens=%d, finish=%s",
        state["attempt_count"],
        result.model,
        result.adapter_id,
        result.token_count,
        result.finish_reason,
    )

    return {"generated_code": extracted, "finish_reason": result.finish_reason}


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

    backend = get_sandbox_backend()
    result = backend.run(script, timeout=timeout)

    stdout = result.stdout
    stderr = result.stderr
    exit_code = result.exit_code

    tests_passed = exit_code == 0 and not result.is_timed_out

    _passed_count, total_count = count_test_results(stdout, stderr)
    tests_ran = total_count > 0

    logger.info(
        "execute_node: exit_code=%d, tests_passed=%s, test_count=%d, tests_ran=%s",
        exit_code,
        tests_passed,
        total_count,
        tests_ran,
    )

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "tests_passed": tests_passed,
        "test_count": total_count,
        "tests_ran": tests_ran,
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

    # Use list concatenation (not .append()) — LangGraph requires immutable state
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
