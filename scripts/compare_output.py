"""Compare gemma base model output vs hypernetwork-adapted output.

Runs the same data_layer subtask prompt through:
  1. Base gemma-2-2b-it (no adapter)
  2. gemma-2-2b-it + methodology adapter from Sakana hypernetwork
  3. gemma-2-2b-it + subtask-specific adapter (error-conditioned)

Then executes each output and reports results.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()

from shared.hardware import get_best_device

DEVICE = get_best_device()
MODEL_NAME = "google/gemma-2-2b-it"

SUBTASK_PROMPT = (
    "From this project spec, implement ONLY the data layer:\n\n"
    "Build a complete Python CLI task manager application. Requirements:\n\n"
    "1. Data layer: TaskStore class backed by SQLite via the stdlib sqlite3 "
    "module. Schema: tasks(id INTEGER PRIMARY KEY, title TEXT NOT NULL, "
    "description TEXT, status TEXT DEFAULT 'todo', priority INTEGER DEFAULT 0, "
    "created_at TEXT, due_date TEXT, tags TEXT). Supports CRUD, filtering by "
    "status/priority/tag, and sorting.\n\n"
    "2. Domain layer: Task dataclass with validation — title must be non-empty, "
    "status must be one of ('todo','in_progress','done','blocked'), priority 0-5, "
    "tags stored as comma-separated string. TaskService class that enforces "
    "business rules: cannot transition from 'done' to 'todo', blocked tasks "
    "cannot move to 'done' directly, and overdue tasks are auto-flagged.\n\n"
    "Focus on: data classes/models, storage class, CRUD operations. "
    "Include basic unit tests for the data layer only. "
    "Use dataclasses, type annotations, and unittest.TestCase."
)

SYSTEM_PROMPT = "You are a Python code generator. Output only code, no explanation."


def run_code(code: str) -> tuple[bool, str, str]:
    """Execute code in subprocess, return (passed, stdout, stderr)."""
    from shared.sandbox import get_sandbox_backend

    backend = get_sandbox_backend()
    result = backend.run(code, timeout=30)
    return (
        result.exit_code == 0 and not result.timed_out,
        result.stdout,
        result.stderr,
    )


async def main() -> None:
    os.environ["INFERENCE_PROVIDER"] = "transformers"
    os.environ["TRANSFORMERS_MODEL_NAME"] = MODEL_NAME
    os.environ["TRANSFORMERS_DEVICE"] = DEVICE

    from inference import get_provider
    from inference.factory import _clear_cache

    _clear_cache()
    provider = get_provider()

    full_prompt = f"{SYSTEM_PROMPT}\n\n{SUBTASK_PROMPT}"

    # --- Run 1: Base model (no adapter) ---
    print("=" * 70)
    print("  RUN 1: Base gemma-2-2b-it (NO adapter)")
    print("=" * 70)
    result_base = await provider.generate(
        prompt=full_prompt,
        model=MODEL_NAME,
        adapter_id=None,
        max_tokens=4096,
    )
    print(f"\n  Tokens generated: {result_base.token_count}")
    print(f"  Code length: {len(result_base.text)} chars")

    import re

    match = re.search(r"```python\s*(.*?)```", result_base.text, re.DOTALL)
    base_code = match.group(1).strip() if match else result_base.text.strip()

    passed, stdout, stderr = run_code(base_code)
    print(f"  Tests passed: {passed}")
    if stderr:
        # Show last 10 lines of stderr
        err_lines = stderr.strip().splitlines()[-10:]
        print("  Stderr (last 10 lines):")
        for line in err_lines:
            print(f"    {line}")

    print("\n  --- Generated Code (first 80 lines) ---")
    for i, line in enumerate(base_code.splitlines()[:80], 1):
        print(f"  {i:3d} | {line}")

    # --- Run 2: With methodology adapter from Sakana ---
    print("\n" + "=" * 70)
    print("  RUN 2: gemma-2-2b-it + Sakana methodology adapter")
    print("=" * 70)

    import tempfile

    from model_training.sakana_d2l import generate_adapter_from_sakana

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate methodology adapter
        methodology_path = Path(__file__).parent.parent / "docs" / "rune-methodology.md"
        methodology = (
            methodology_path.read_text()
            if methodology_path.exists()
            else "coding standards"
        )

        bootstrap_text = (
            f"PROJECT: {SUBTASK_PROMPT[:500]}\n\n"
            f"METHODOLOGY:\n{methodology}\n\n"
            "PHASE: bootstrap — apply methodology to plan and implement this project."
        )

        adapter_path = generate_adapter_from_sakana(
            text=bootstrap_text,
            output_dir=os.path.join(tmpdir, "methodology"),
            device=DEVICE,
        )
        await provider.load_adapter("methodology_compare", adapter_path)

        result_adapted = await provider.generate(
            prompt=full_prompt,
            model=MODEL_NAME,
            adapter_id="methodology_compare",
            max_tokens=4096,
        )
        print(f"\n  Tokens generated: {result_adapted.token_count}")
        print(f"  Code length: {len(result_adapted.text)} chars")

        match2 = re.search(r"```python\s*(.*?)```", result_adapted.text, re.DOTALL)
        adapted_code = (
            match2.group(1).strip() if match2 else result_adapted.text.strip()
        )

        passed2, stdout2, stderr2 = run_code(adapted_code)
        print(f"  Tests passed: {passed2}")
        if stderr2:
            err_lines2 = stderr2.strip().splitlines()[-10:]
            print("  Stderr (last 10 lines):")
            for line in err_lines2:
                print(f"    {line}")

        print("\n  --- Generated Code (first 80 lines) ---")
        for i, line in enumerate(adapted_code.splitlines()[:80], 1):
            print(f"  {i:3d} | {line}")

        # --- Run 3: With error-conditioned adapter (simulate retry) ---
        print("\n" + "=" * 70)
        print("  RUN 3: gemma-2-2b-it + error-conditioned adapter (retry)")
        print("=" * 70)

        if not passed2:
            # Build error trajectory

            error_text = stderr2[:500] if stderr2 else "no test output"
            error_trajectory = (
                f"SUBTASK: 1/4 — data_layer\n"
                f"FOCUS: data model CRUD and queries\n\n"
                f"TESTS: 0/0 FAILING\n\n"
                f"ERRORS:\n{error_text}\n\n"
                f"FIX: Check syntax, imports, indentation. Ensure all names defined."
            )

            adapter_path2 = generate_adapter_from_sakana(
                text=error_trajectory,
                output_dir=os.path.join(tmpdir, "error_conditioned"),
                device=DEVICE,
            )
            await provider.unload_adapter("methodology_compare")
            await provider.load_adapter("error_conditioned", adapter_path2)

            result_retry = await provider.generate(
                prompt=full_prompt,
                model=MODEL_NAME,
                adapter_id="error_conditioned",
                max_tokens=4096,
            )
            print(f"\n  Tokens generated: {result_retry.token_count}")
            print(f"  Code length: {len(result_retry.text)} chars")

            match3 = re.search(r"```python\s*(.*?)```", result_retry.text, re.DOTALL)
            retry_code = (
                match3.group(1).strip() if match3 else result_retry.text.strip()
            )

            passed3, stdout3, stderr3 = run_code(retry_code)
            print(f"  Tests passed: {passed3}")
            if stderr3:
                err_lines3 = stderr3.strip().splitlines()[-10:]
                print("  Stderr (last 10 lines):")
                for line in err_lines3:
                    print(f"    {line}")

            print("\n  --- Generated Code (first 80 lines) ---")
            for i, line in enumerate(retry_code.splitlines()[:80], 1):
                print(f"  {i:3d} | {line}")
        else:
            print("\n  (Skipped — Run 2 already passed)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"  Base model (no adapter):     tests_passed={passed}, code_len={len(base_code)}"
    )
    print(
        f"  Methodology adapter:         tests_passed={passed2}, code_len={len(adapted_code)}"
    )
    if not passed2:
        print(
            f"  Error-conditioned adapter:   tests_passed={passed3}, code_len={len(retry_code)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
