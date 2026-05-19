"""Gate 2: Multi-benchmark robustness (Table 3).

Runs all 6 REQUIRED_BENCHMARKS for both substrate baseline (Qwen 3.5 +
DeltaCoder) and Rune adapter (substrate + hypernetwork adapter), then
applies the strict gate from round2_gate.py.

Supports two modes for the Rune condition:
  --rune-adapter-dir  Use pre-generated adapters (two-phase GPU sharing).
  --hypernet-checkpoint  Live hypernetwork (requires exclusive GPU).

Usage:
    uv run python scripts/paper/run_gate2.py \
        --rune-adapter-dir evaluation_results/paper/rune_adapters \
        --output evaluation_results/gate2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def _build_pregenerated_adapter_generator(
    adapter_dir: str,
    benchmarks: list[str],
) -> Any:
    """Build an adapter_generator from pre-generated adapters on disk."""
    from evaluation.benchmarks.runner import _ADAPTER_REGISTRY, _import_adapter

    manifest_path = Path(adapter_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    prompt_to_adapter: dict[str, str] = {}
    for bench_id in benchmarks:
        if bench_id not in manifest:
            logger.warning("No pre-generated adapters for %s", bench_id)
            continue
        pid_map = manifest[bench_id]
        adapter = _import_adapter(_ADAPTER_REGISTRY[bench_id])
        problems = adapter.load_problems(max_samples=None, seed=42)
        for problem in problems:
            if problem.problem_id in pid_map:
                prompt_to_adapter[problem.prompt] = pid_map[problem.problem_id]
    logger.info("Loaded %d pre-generated adapter paths", len(prompt_to_adapter))

    def adapter_generator(prompt: str) -> str | None:
        return prompt_to_adapter.get(prompt)

    return adapter_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2 evaluation")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter",
        default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument(
        "--hypernet-checkpoint",
        type=str,
        default=None,
        help="Path to trained hypernetwork checkpoint (live mode)",
    )
    parser.add_argument(
        "--rune-adapter-dir",
        type=str,
        default=None,
        help="Path to pre-generated adapter dir (with manifest.json)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/gate2.json")
    )
    args = parser.parse_args()

    if not args.rune_adapter_dir and not args.hypernet_checkpoint:
        parser.error("Either --rune-adapter-dir or --hypernet-checkpoint is required")

    import subprocess

    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from inference.factory import get_provider
    from model_training.round2_gate import (
        REQUIRED_BENCHMARKS,
        evaluate_round2_gate,
    )
    from model_training.training_common import (
        mlflow_log_artifact,
        mlflow_log_params,
        setup_mlflow,
    )

    mlflow_ok = setup_mlflow("paper-gate2", tracking_uri=None)
    if mlflow_ok:
        import mlflow

        mlflow.start_run(run_name="gate2")
        mlflow_log_params(
            {
                "model": args.model,
                "warm_start_adapter": args.warm_start_adapter,
                "hypernet_checkpoint": args.hypernet_checkpoint or "pregenerated",
                "rune_adapter_dir": args.rune_adapter_dir or "",
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
            }
        )

    benchmarks = list(REQUIRED_BENCHMARKS)
    provider = get_provider()

    print("=== Gate 2: Substrate baseline (Qwen 3.5 + DeltaCoder) ===")
    baseline_stack = AdapterStack(
        base_model=args.model,
        adapter_ids=[args.warm_start_adapter],
        adapter_paths={},
        provider=provider,
    )

    def _on_verdict(
        bench_id: str,
        verdict: Any,
        running_p1: float,
        n_done: int,
        n_total: int,
    ) -> None:
        if mlflow_ok:
            mlflow.log_metric(f"substrate_{bench_id}_running", running_p1, step=n_done)
        if n_done % 50 == 0 or n_done == n_total:
            print(
                f"  [substrate/{bench_id}] {n_done}/{n_total} Pass@1={running_p1:.2%}"
            )

    baseline_scores: dict[str, float] = {}
    for bench in benchmarks:
        result = run_benchmark(baseline_stack, bench, on_verdict=_on_verdict)
        baseline_scores[bench] = result.pass_at_1
        print(f"  [substrate] {bench}: {result.pass_at_1:.2%}")

    print("\n=== Gate 2: Rune (substrate + hypernetwork adapter) ===")

    if args.rune_adapter_dir:
        adapter_generator = _build_pregenerated_adapter_generator(
            args.rune_adapter_dir, benchmarks
        )
    else:
        import tempfile

        from rune_runner import run_hypernetwork  # type: ignore[import-not-found]

        tmp_dir = tempfile.mkdtemp(prefix="rune_gate2_")

        def adapter_generator(prompt: str) -> str | None:
            return run_hypernetwork(
                trajectory_text=prompt,
                output_dir=tmp_dir,
                base_model_id=args.model,
                checkpoint_path=args.hypernet_checkpoint,
                device=args.device,
            )

    def _on_verdict_rune(
        bench_id: str,
        verdict: Any,
        running_p1: float,
        n_done: int,
        n_total: int,
    ) -> None:
        if mlflow_ok:
            mlflow.log_metric(f"rune_{bench_id}_running", running_p1, step=n_done)
        if n_done % 50 == 0 or n_done == n_total:
            print(f"  [rune/{bench_id}] {n_done}/{n_total} Pass@1={running_p1:.2%}")

    rune_stack = AdapterStack(
        base_model=args.model,
        adapter_ids=[args.warm_start_adapter],
        adapter_paths={},
        provider=provider,
        adapter_generator=adapter_generator,
    )
    rune_scores: dict[str, float] = {}
    for bench in benchmarks:
        result = run_benchmark(rune_stack, bench, on_verdict=_on_verdict_rune)
        rune_scores[bench] = result.pass_at_1
        print(f"  [rune] {bench}: {result.pass_at_1:.2%}")
        if mlflow_ok:
            mlflow.log_metric(f"rune_{bench}", rune_scores[bench])

    scores_input = {
        bench: {"baseline": baseline_scores[bench], "round2": rune_scores[bench]}
        for bench in benchmarks
    }
    gate_result = evaluate_round2_gate(scores_input)

    report: dict[str, Any] = {
        "baseline_scores": baseline_scores,
        "rune_scores": rune_scores,
        "gate_result": gate_result,
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "hypernet_checkpoint": args.hypernet_checkpoint or "pregenerated",
        "rune_adapter_dir": args.rune_adapter_dir,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    verdict = "PASS" if gate_result["passed"] else "FAIL"
    print(f"\nGate 2 verdict: {verdict}")
    print(f"Output: {args.output}")

    if mlflow_ok:
        for bench in benchmarks:
            mlflow.log_metric(f"substrate_{bench}", baseline_scores[bench])
            mlflow.log_metric(f"rune_{bench}", rune_scores[bench])
        mlflow.log_metric("gate2_passed", 1.0 if gate_result["passed"] else 0.0)
        mlflow_log_artifact(str(args.output))
        mlflow.end_run()


if __name__ == "__main__":
    main()
