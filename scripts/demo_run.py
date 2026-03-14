"""Demo: run a simple coding problem through the Rune agent loop.

Uses Ollama with qwen2.5-coder:1.5b to solve a coding task,
exercising the full generate → execute → reflect → retry loop.

Usage:
    INFERENCE_PROVIDER=ollama RUNE_MODEL=qwen2.5-coder:1.5b uv run scripts/demo_run.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

# Configure before imports
os.environ.setdefault("INFERENCE_PROVIDER", "ollama")
os.environ.setdefault("RUNE_MODEL", "qwen2.5-coder:1.5b")
os.environ.setdefault("RUNE_EXEC_TIMEOUT", "30")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

from rune_agent.graph import create_graph  # noqa: E402
from rune_agent.state import RuneState  # noqa: E402

TASK = {
    "task_description": (
        "Write a Python function called `two_sum` that takes a list of integers `nums` "
        "and an integer `target`, and returns a list of two indices such that the "
        "numbers at those indices add up to `target`. You may assume each input has "
        "exactly one solution, and you may not use the same element twice."
    ),
    "test_suite": "\n".join(
        [
            "# Tests for two_sum",
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]",
            "assert two_sum([3, 2, 4], 6) == [1, 2]",
            "assert two_sum([3, 3], 6) == [0, 1]",
            "print('All tests passed!')",
        ]
    ),
}


async def main() -> dict:
    model = os.environ["RUNE_MODEL"]
    provider = os.environ["INFERENCE_PROVIDER"]
    logger.info("Model: %s via %s", model, provider)

    # Build initial state
    state: RuneState = {
        "task_description": TASK["task_description"],
        "task_type": "function",
        "test_suite": TASK["test_suite"],
        "adapter_ids": [],
        "session_id": str(uuid.uuid4()),
        "attempt_count": 0,
        "max_attempts": 5,
        "generated_code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "tests_passed": False,
        "trajectory": [],
        "outcome": None,
    }

    logger.info("Task: %s", TASK["task_description"][:80] + "...")
    logger.info("Running agent loop (max %d attempts)...", state["max_attempts"])

    graph = create_graph()
    final_state = await graph.ainvoke(state)

    # Report results
    outcome = final_state["outcome"]
    attempts = final_state["attempt_count"]

    print("\n" + "=" * 60)
    print(f"OUTCOME: {outcome}")
    print(f"ATTEMPTS: {attempts}")
    print("=" * 60)

    for i, step in enumerate(final_state["trajectory"]):
        status = "PASS" if step["tests_passed"] else "FAIL"
        print(f"\n--- Attempt {i + 1} [{status}] ---")
        print(f"Code:\n{step['generated_code']}")
        if step["stderr"]:
            print(f"Stderr:\n{step['stderr'][:500]}")
        if step["stdout"]:
            print(f"Stdout:\n{step['stdout'][:500]}")

    print("\n" + "=" * 60)
    if outcome == "success":
        print("The model solved the problem!")
    else:
        print("The model did not solve the problem within the attempt limit.")

    return final_state


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result["outcome"] == "success" else 1)
