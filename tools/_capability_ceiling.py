"""Capability ceiling: base model, NO adapter, single-shot, NO engine loop.

REMOVE-BEFORE-MERGE. The denominator for engine-vs-capability attribution (issue
#52): "can the model solve this at all?" with nothing but a direct prompt + the
full spec. No decompose/plan/diagnose/repair/integrate, no adapter (scale 0 ==
base). A task the base model cannot one-shot here is a capability limit, not an
engine bug; a task it CAN one-shot but the engine+adapter fails is an engine bug.

Run: tools/run_guarded.sh /tmp/goal3/ceiling.log tools/_capability_ceiling.py \
       --tasks benchmarks/goal3_multistep_all8.json --out /tmp/goal3/ceiling.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"

_PROMPT = """\
Write a single Python function that solves the task below. Output only the function
in one ```python code block — no explanation, no tests.

{description}

Define the function named exactly `{entry_point}`."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint", default=C3_CKPT)
    args = ap.parse_args()

    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.continuation import (  # noqa: PLC0415
        extract_partial_code,
        strip_self_tests,
    )
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415

    tasks = json.loads(Path(args.tasks).read_text())
    if args.limit:
        tasks = tasks[: args.limit]

    # adapter_scaling=0 -> lora_B zeroed == base model. No adapter is generated or
    # hot-swapped below: we call generate() on the untouched base weights.
    cfg = load_rune_config(None).override(
        checkpoint_path=args.checkpoint,
        adapter_scaling=0.0,
        seed=0,
        model_judge=False,
    )
    model = ModelWrapper.from_config(cfg)
    rc = cfg.to_dict()

    results = []
    for t in tasks:
        prompt = _PROMPT.format(
            description=t["description"], entry_point=t["entry_point"]
        )
        gen = asyncio.run(
            model.generate(
                prompt=prompt,
                max_tokens=rc.get("max_tokens", 2048),
                temperature=rc.get("temperature", 0.3),
                repetition_penalty=rc.get("repetition_penalty", 1.1),
                top_p=rc.get("top_p", 0.9),
                no_repeat_ngram_size=rc.get("no_repeat_ngram_size", 0),
                presence_penalty=rc.get("presence_penalty", 0.0),
                thinking_budget=rc.get("thinking_budget", 0),
            )
        )
        code = extract_partial_code(gen.text)
        full = strip_self_tests(code) + "\n\n" + t["test_code"]
        sb = run_in_sandbox(full)
        passed = sb.exit_code == 0
        results.append(
            {
                "task_id": t["task_id"],
                "passed": passed,
                "code": code,
                "stderr": (sb.stderr or "")[:600] if not passed else "",
            }
        )
        print(
            f"{'PASS' if passed else 'fail'} {t['task_id']} (code {len(code)} chars)",
            flush=True,
        )

    n_pass = sum(r["passed"] for r in results)
    out = {
        "arm": "capability_ceiling_base_no_adapter",
        "pass_at_1": n_pass / len(results) if results else 0.0,
        "passed_tasks": n_pass,
        "total_tasks": len(results),
        "per_task": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(
        f"\nCAPABILITY CEILING (base, no adapter): {n_pass}/{len(results)}", flush=True
    )


if __name__ == "__main__":
    main()
