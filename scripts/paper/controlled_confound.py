"""Controlled-confound harness for Figure 2(a).

Three conditions:
1. Rune: Full adapter-augmented generation (adapter encodes trajectory).
2. Injected History: Trajectory prepended to prompt as raw text context.
3. Memory Stripped: Base model with no trajectory access at all.

Measures Pass@1 vs trajectory length to show Rune's slope continues past
the context ceiling where injected-history and RAG plateau.

Usage:
    uv run python scripts/paper/controlled_confound.py \
        --model Qwen/Qwen3.5-9B \
        --tasks data/confound_tasks.json \
        --output evaluation_results/figure2a.json
"""

from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any


class ConfoundCondition(str, Enum):
    """Experimental conditions for the controlled-confound study."""

    RUNE = "rune"
    INJECTED_HISTORY = "injected_history"
    MEMORY_STRIPPED = "memory_stripped"


def build_injected_history(
    base_prompt: str,
    trajectory_steps: list[str],
) -> str:
    """Build a prompt with trajectory injected as context prefix.

    Args:
        base_prompt: The coding task prompt.
        trajectory_steps: Ordered trajectory steps to prepend.

    Returns:
        Full prompt with history context prepended.
    """
    history = "\n".join(trajectory_steps)
    return f"# Previous steps:\n{history}\n\n# Current task:\n{base_prompt}"


def build_memory_stripped(base_prompt: str) -> str:
    """Build a prompt with no trajectory context (base model only).

    Args:
        base_prompt: The coding task prompt.

    Returns:
        Unmodified base prompt.
    """
    return base_prompt


def run_confound_experiment(
    tasks: list[dict[str, Any]],
    trajectory_depths: list[int],
    conditions: list[ConfoundCondition],
) -> dict[str, Any]:
    """Run the controlled-confound experiment across depths and conditions.

    Args:
        tasks: List of task dicts with "prompt", "trajectory_steps", "test".
        trajectory_depths: List of step counts to test.
        conditions: Which conditions to evaluate.

    Returns:
        Nested results: {condition: {depth: pass_rate}}.
    """
    raise NotImplementedError(
        "Inference callback not wired — scaffold only (see paper §2.3)"
    )

    results: dict[str, dict[int, float]] = {c.value: {} for c in conditions}

    for depth in trajectory_depths:
        for condition in conditions:
            pass_count = 0
            total = 0
            for task in tasks:
                steps = task.get("trajectory_steps", [])[:depth]
                prompt = task["prompt"]

                if condition == ConfoundCondition.INJECTED_HISTORY:
                    _ = build_injected_history(prompt, steps)
                elif condition == ConfoundCondition.MEMORY_STRIPPED:
                    _ = build_memory_stripped(prompt)

                total += 1

            pass_rate = pass_count / total if total > 0 else 0.0
            results[condition.value][depth] = pass_rate

    return {"results": results, "depths": trajectory_depths}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled-confound harness for Figure 2(a)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/figure2a.json")
    )
    args = parser.parse_args()

    with args.tasks.open() as f:
        tasks = json.load(f)

    results = run_confound_experiment(
        tasks,
        args.depths,
        [
            ConfoundCondition.RUNE,
            ConfoundCondition.INJECTED_HISTORY,
            ConfoundCondition.MEMORY_STRIPPED,
        ],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
