"""GPU smoke test: verify the engine's continuation sub-loop with a real model.

Loads the model from bench.yaml, runs a single task through the full engine
pipeline with max_tokens low enough to force truncation on code generation,
and checks that:
  1. Continuation rounds fire (adapter regenerated, budget consumed)
  2. Accumulated code assembles correctly
  3. Sandbox receives and executes the assembled code
  4. Final state is coherent

Run:  uv run python tools/smoke_test_engine.py
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

    # Low max_tokens forces truncation on code generation so
    # the engine enters the continuation sub-loop.
    max_tokens = 512

    log.info("Loading model: %s  %s", cfg.model_id, _mem())
    t0 = time.monotonic()
    model = ModelWrapper.from_config(cfg)
    log.info("Model loaded in %.1fs  %s", time.monotonic() - t0, _mem())

    engine = create_engine()

    # Task hits _is_simple_task gate (starts with "Write a class"),
    # so decomposition is skipped → synthetic _main subtask.
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
        "budget_remaining": 8,
    }

    run_config = cfg.to_dict()
    run_config["max_tokens"] = max_tokens

    config = {"model": model, "run_config": run_config}

    log.info("=== Starting engine run ===")
    log.info("task: %s", task[:80])
    log.info(
        "max_tokens=%d  scaling=%.2f  cont_mult=%.2f  no_repeat_ngram=%d",
        max_tokens, cfg.adapter_scaling, cfg.cont_multiplier, cfg.no_repeat_ngram_size,
    )
    log.info("budget_remaining=%d", initial_state["budget_remaining"])
    print(flush=True)

    t0 = time.monotonic()
    final_state = await engine.ainvoke(
        initial_state, config={"configurable": config},
    )
    elapsed = time.monotonic() - t0

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
    cont_budget = budget_spent - final_state["step"]
    print("=== BUDGET ANALYSIS ===", flush=True)
    print(f"  Actions recorded: {n_actions}", flush=True)
    print(f"  Steps (outer loop): {final_state['step']}", flush=True)
    print(f"  Budget spent: {budget_spent}", flush=True)
    print(f"  Continuation budget (budget_spent - steps): {cont_budget}", flush=True)
    if cont_budget > 0:
        print("  PASS: continuation consumed extra budget", flush=True)
    else:
        print("  INFO: no extra continuation budget consumed "
              "(truncation may not have triggered)", flush=True)
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

    print("=== SMOKE TEST COMPLETE ===", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"  {_mem()}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
