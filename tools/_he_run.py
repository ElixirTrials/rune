"""Run HumanEval+ (EvalPlus) through the rune engine: base / c3 / scale0 arms.

Corroborates the LCB uplift on a second function-level benchmark. Both arms are
graded by the rune sandbox against the EvalPlus hardened ("plus") tests, so the
comparison is apples-to-apples. base = single-shot zero-shot (no engine loop);
c3 = escalate adapter@1.0; scale0 = escalate adapter-off. Emits {task_id: passed}.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
ARMS = {"base": 0.0, "scale0": 0.0, "c3": 1.0}


def load_he_tasks() -> list[Any]:
    import datasets as hf  # noqa: PLC0415

    from rune.bench.runner import BenchTask  # noqa: PLC0415

    ds = hf.load_dataset("evalplus/humanevalplus", split="test")
    tasks = []
    for r in ds:
        ep = r["entry_point"]
        test_code = r["test"] + f"\n\ncheck({ep})\n"
        tasks.append(
            BenchTask(
                task_id=r["task_id"],
                description=r["prompt"],
                test_code=test_code,
                entry_point=ep,
                signature="",
                public_checks="",  # docstring >>> examples -> doctest oracle fallback
            )
        )
    return tasks


async def _gen_base(model: Any, tasks: list[Any], cfg: Any) -> dict[str, bool]:
    import torch  # noqa: PLC0415

    from rune.engine.continuation import (  # noqa: PLC0415
        extract_partial_code,
        strip_self_tests,
    )
    from rune.engine.parse import render_template  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415

    rc = cfg.to_dict()
    seed = rc.get("seed")
    out: dict[str, bool] = {}
    for i, t in enumerate(tasks):
        if seed is not None:
            torch.manual_seed(seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + i)
        prompt = render_template(
            "prompt_zeroshot", task_description=t.description, entry_point=t.entry_point
        )
        gen = await model.generate(
            prompt=prompt,
            system_prompt="You are a code generator.",
            output_schema=None,
            max_tokens=rc.get("max_tokens", 2048),
            temperature=rc.get("temperature", 0.3),
            repetition_penalty=rc.get("repetition_penalty", 1.1),
            top_p=rc.get("top_p", 0.9),
            no_repeat_ngram_size=rc.get("no_repeat_ngram_size", 0),
            presence_penalty=rc.get("presence_penalty", 0.0),
            thinking_budget=rc.get("thinking_budget", 0),
        )
        code = extract_partial_code(gen.text)
        full = strip_self_tests(code) + "\n\n" + t.test_code
        passed = run_in_sandbox(full, timeout=30).exit_code == 0
        out[t.task_id] = passed
        print(f"base {t.task_id}: {'PASS' if passed else 'fail'} ({len(code)}c)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=24)
    ap.add_argument("--prompt-mode", default="escalate")
    ap.add_argument("--experiment", default="issue52-humanevalplus")
    args = ap.parse_args()

    import mlflow  # noqa: PLC0415

    from rune.bench.runner import run_benchmark  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT,
        adapter_scaling=ARMS[args.arm],
        seed=args.seed,
        max_phase_iterations=args.max_iters,
        prompt_mode=args.prompt_mode,
    )
    model = ModelWrapper.from_config(cfg)
    tasks = load_he_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    ckpt_sha = hashlib.sha256(Path(C3_CKPT).read_bytes()).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    ).stdout.strip()
    configure_mlflow(args.experiment)
    params = {
        **cfg.to_dict(),
        "arm": args.arm,
        "benchmark": "humanevalplus",
        "n_tasks": len(tasks),
        "checkpoint_sha256": ckpt_sha,
        "engine_commit": commit,
    }
    out_path = Path(args.out)
    with tracked_run(f"he-{args.arm}-seed{args.seed}", params=params):
        if args.arm == "base":
            perq = asyncio.run(_gen_base(model, tasks, cfg))
        else:
            engine = create_engine()
            config: dict[str, Any] = {
                "model": model,
                "run_config": cfg.to_dict(),
                "benchmark": "humanevalplus",
            }
            sess = out_path.with_suffix("")  # per-task sessions dir for progress
            sess.mkdir(parents=True, exist_ok=True)
            result = asyncio.run(run_benchmark(tasks, engine, config, sessions_dir=sess))
            perq = {tr.task_id: bool(tr.passed) for tr in result.per_task}
        npass = sum(perq.values())
        n = len(perq)
        mlflow.log_metric("pass_at_1", npass / n if n else 0.0)
        mlflow.log_metric("passed_tasks", npass)
        mlflow.log_metric("total_tasks", n)
        out_path.write_text(json.dumps(perq, indent=1))
        mlflow.log_artifact(str(out_path))
    print(f"{args.arm}: {npass}/{n} = {npass / n:.3f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
