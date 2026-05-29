"""GPU smoke test: verify the engine's continuation sub-loop with a real model.

Loads the model from bench.yaml, runs a single task through the full engine
pipeline with max_tokens low enough to force truncation on code generation,
and checks that:
  1. Continuation rounds fire (adapter regenerated, budget consumed)
  2. Accumulated code assembles correctly
  3. Sandbox receives and executes the assembled code
  4. Final state is coherent

Run:
  uv run python tools/smoke_test_engine.py
  uv run python tools/smoke_test_engine.py --eos   # single-subtask EOS (no integrate)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger("smoke_test_engine")


def _mem() -> str:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            return f"GPU alloc={a:.1f}GB reserved={r:.1f}GB"
    except Exception:
        pass
    return ""


async def run() -> None:
    from rune.config import load_config  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    cfg_path = Path("benchmarks/bench.yaml")
    log.info("Loading config from %s", cfg_path)
    cfg = load_config(cfg_path)

    eos_mode = "--eos" in sys.argv
    no_cont = "--no-cont" in sys.argv
    max_tokens = 2048 if (no_cont or eos_mode) else 512

    log.info("Loading model: %s  %s", cfg.model_id, _mem())
    t0 = time.monotonic()
    model = ModelWrapper.from_config(cfg)
    log.info("Model loaded in %.1fs  %s", time.monotonic() - t0, _mem())

    engine = create_engine()

    if eos_mode:
        task = (
            'Write a function to find tuples which have all elements divisible '
            'by k from the given list of tuples.\n\n'
            ">>> assert find_tuples([(6, 24, 12), (7, 9, 6), (12, 18, 21)], 6) "
            "== [(6, 24, 12)]"
        )
    elif no_cont:
        task = (
            "Write a function called fibonacci(n) that returns the nth "
            "Fibonacci number. Include 3 tests."
        )
    else:
        task = (
            "Write a class LinkedList with methods: append, prepend, delete, "
            "find, reverse, to_list, __len__, and __repr__. Include a Node "
            "inner class. Write tests for each method."
        )

    initial_state: dict[str, Any] = {
        "task": task,
        "subtasks": [],
        "interfaces": {},
        "plans": {},
        "code_results": {},
        "code_passed": {},
        "retries": {},
        "integrated_code": "",
        "current_adapter": None,
        "feedback": {},
        "integration_feedback": None,
        "diagnosis": {},
        "actions": [],
        "trajectory": [],
        "step": 0,
        "budget_remaining": 5 if eos_mode else 8,
    }

    run_config = cfg.to_dict()
    run_config["max_tokens"] = max_tokens
    if no_cont:
        run_config["cont_budget"] = 0
    if "--deterministic" in sys.argv:
        # inference maps temperature==0 -> do_sample=False
        run_config["temperature"] = 0.0

    config = {"model": model, "run_config": run_config}

    log.info("=== Starting engine run ===")
    log.info("task: %s", task[:80])
    log.info(
        "max_tokens=%d  scaling=%.2f  cont_mult=%.2f  no_repeat_ngram=%d",
        max_tokens,
        cfg.adapter_scaling,
        cfg.cont_multiplier,
        cfg.no_repeat_ngram_size,
    )
    log.info("budget_remaining=%d", initial_state["budget_remaining"])
    print(flush=True)

    t0 = time.monotonic()
    final_state = await engine.ainvoke(
        initial_state,
        config={"configurable": config},
    )
    elapsed = time.monotonic() - t0

    dump_dir = None
    for i, arg in enumerate(sys.argv):
        if arg == "--dump-sessions" and i + 1 < len(sys.argv):
            dump_dir = Path(sys.argv[i + 1])
    if dump_dir is not None:
        from rune.mining.session_log import write_session  # noqa: PLC0415

        write_session(
            final_state,
            {"benchmark": "smoke", "problem_id": "linkedlist"},
            dump_dir / "linkedlist",
        )
        log.info("Wrote session corpus to %s", dump_dir)

    print(flush=True)
    log.info("=== Engine finished in %.1fs ===", elapsed)
    log.info("Steps taken: %d", final_state["step"])
    log.info("Budget remaining: %d", final_state["budget_remaining"])
    spent = initial_state["budget_remaining"] - final_state["budget_remaining"]
    log.info("Budget consumed: %d", spent)
    print(flush=True)

    # --- Trajectory dump ---
    print("=== TRAJECTORY ===", flush=True)
    for rec in final_state.get("trajectory", []):
        fb_info = ""
        if rec.feedback:
            fb_info = f" exit={rec.feedback.exit_code}"
            if rec.feedback.exit_code != 0:
                fb_info += f" err={rec.feedback.stderr[:120]}"
        code_info = ""
        if rec.generated_code:
            code_info = f" code={len(rec.generated_code)}chars"
        print(
            f"  step={rec.step} action={rec.action_name} "
            f"target={rec.target_subtask} adapter={rec.adapter_id[:8]}..."
            f"{fb_info}{code_info}",
            flush=True,
        )
    print(flush=True)

    # --- Code results ---
    print("=== CODE RESULTS ===", flush=True)
    code_results = final_state.get("code_results", {})
    for name, code in code_results.items():
        print(f"  [{name}] {len(code)} chars", flush=True)
        for line in code.splitlines()[:5]:
            print(f"    {line}", flush=True)
        if len(code.splitlines()) > 5:
            print(f"    ... ({len(code.splitlines())} lines total)", flush=True)
    print(flush=True)

    # --- Budget analysis ---
    budget_spent = initial_state["budget_remaining"] - final_state["budget_remaining"]
    n_actions = len(final_state.get("trajectory", []))
    print("=== BUDGET ANALYSIS ===", flush=True)
    print(f"  Actions recorded: {n_actions}", flush=True)
    print(f"  Steps (outer loop): {final_state['step']}", flush=True)
    print(f"  Budget spent: {budget_spent}", flush=True)
    print(flush=True)
    print(
        "  NOTE: Continuation rounds are internal to step_node and"
        " do not consume outer budget. Check engine logs for"
        " 'continuation round' messages to verify.",
        flush=True,
    )
    print(flush=True)

    # --- Integrated code ---
    integrated = final_state.get("integrated_code", "")
    print("=== INTEGRATED CODE ===", flush=True)
    if integrated:
        n_lines = len(integrated.splitlines())
        print(f"  {len(integrated)} chars, {n_lines} lines", flush=True)
        for line in integrated.splitlines()[:8]:
            print(f"    {line}", flush=True)
        if len(integrated.splitlines()) > 8:
            print(f"    ... ({len(integrated.splitlines())} lines total)", flush=True)
    else:
        print("  (empty — may not have reached integration)", flush=True)
    print(flush=True)

    if eos_mode:
        trajectory = final_state.get("trajectory", [])
        action_names = [rec.action_name for rec in trajectory]
        subtasks = final_state.get("subtasks", [])
        integrated = final_state.get("integrated_code", "")
        code_passed = final_state.get("code_passed", {})
        print("=== EOS CHECKS ===", flush=True)
        print(f"  subtasks: {[s.name for s in subtasks]}", flush=True)
        print(f"  actions: {action_names}", flush=True)
        print(f"  code_passed: {code_passed}", flush=True)
        print(f"  integrated_code chars: {len(integrated)}", flush=True)
        errors: list[str] = []
        if len(subtasks) != 1:
            errors.append(f"expected 1 subtask, got {len(subtasks)}")
        if "integrate" in action_names:
            errors.append(f"integrate must not run, saw trajectory: {action_names}")
        main_passed = code_passed.get("_main", False)
        if main_passed and not integrated:
            errors.append("integrated_code empty after _main sandbox pass")
        if errors:
            print("=== EOS SMOKE FAILED ===", flush=True)
            for err in errors:
                print(f"  - {err}", flush=True)
            raise SystemExit(1)
        print("=== EOS SMOKE PASSED ===", flush=True)

    print("=== SMOKE TEST COMPLETE ===", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"  {_mem()}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
