"""Run the Rune pipeline on ONE MBPP problem — no exception swallowing.

Errors propagate with full tracebacks so they can be fixed before HPO.

Usage:
    uv run python scripts/optimization/run_single_mbpp.py \
        --hypernet-checkpoint s3://.../checkpoint.pt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

from evaluation.benchmarks.mbpp import MBPPAdapter  # noqa: E402
from evaluation.benchmarks.protocol import Problem  # noqa: E402

logger = logging.getLogger("single_mbpp")

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"


def run_one(
    problem: Problem,
    *,
    hypernet_checkpoint: str,
    base_model: str,
    device: str,
) -> dict:
    import mlflow  # noqa: PLC0415
    from rune_runner import run_phased_pipeline  # type: ignore[import-not-found]

    mlflow.set_tag("problem_id", problem.problem_id)
    mlflow.set_tag("prompt", problem.prompt[:250])

    logger.info(
        "Running problem %s: %s",
        problem.problem_id,
        problem.prompt[:120].replace("\n", " "),
    )
    start = time.time()
    result = asyncio.run(
        run_phased_pipeline(
            project_prompt=problem.prompt,
            checkpoint_path=hypernet_checkpoint,
            base_model_id=base_model,
            device=device,
        )
    )
    elapsed = time.time() - start
    code = result.get("accumulated_code", "")
    verdict = MBPPAdapter().score(problem, code)

    mlflow.log_metrics(
        {
            "verdict/passed": float(verdict.passed),
            "verdict/code_len": float(len(code)),
            "verdict/wall_time_s": round(elapsed, 1),
        }
    )
    if verdict.error:
        mlflow.set_tag("verdict/error", verdict.error[:500])

    logger.info(
        "Done in %.1fs — passed=%s code_len=%d",
        elapsed,
        verdict.passed,
        len(code),
    )
    if not verdict.passed:
        logger.warning("Verdict error: %s", verdict.error)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    import mlflow

    # sagemaker-mlflow plugin unconditionally claims in_context()=True then
    # crashes parsing non-ARN tracking URIs; must suppress after import.
    logging.getLogger("mlflow.tracking.request_header.registry").setLevel(logging.ERROR)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.config.enable_async_logging()
    mlflow.set_experiment("single-mbpp")
    mlflow.start_run(run_name="single-mbpp")
    parser = argparse.ArgumentParser(
        description="Single MBPP problem through Rune pipeline"
    )
    parser.add_argument("--hypernet-checkpoint", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--problem-id",
        default=None,
        help="Specific MBPP problem ID (e.g. 'mbpp/420'). Random if omitted.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    problems = MBPPAdapter().load_problems()
    if args.problem_id:
        matches = [p for p in problems if p.problem_id == args.problem_id]
        if not matches:
            raise SystemExit(f"Problem {args.problem_id!r} not found in MBPP dataset")
        problem = matches[0]
    else:
        import random

        problem = random.Random(args.seed).choice(problems)
        logger.info("Selected random problem: %s", problem.problem_id)

    os.environ.setdefault("INFERENCE_PROVIDER", "transformers")
    mlflow.langchain.autolog(run_tracer_inline=True)
    try:
        run_one(
            problem,
            hypernet_checkpoint=args.hypernet_checkpoint,
            base_model=args.base_model,
            device=args.device,
        )
        mlflow.end_run()
    except BaseException:
        mlflow.set_tag("error", "pipeline_crash")
        mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
