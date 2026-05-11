"""Gate 2: Multi-benchmark robustness (Table 3).

Runs all 6 REQUIRED_BENCHMARKS for both substrate baseline (Qwen 3.5 +
DeltaCoder) and Rune adapter (substrate + hypernetwork adapter), then
applies the strict gate from round2_gate.py.

Usage:
    uv run python scripts/paper/run_gate2.py \
        --hypernet-checkpoint path/to/checkpoint.bin \
        --output evaluation_results/gate2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path

setup_path()

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


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
        required=True,
        help="Path to trained hypernetwork checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/gate2.json")
    )
    args = parser.parse_args()

    import subprocess
    import tempfile

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
                "hypernet_checkpoint": args.hypernet_checkpoint,
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
    baseline_scores: dict[str, float] = {}
    for bench in benchmarks:
        result = run_benchmark(baseline_stack, bench)
        baseline_scores[bench] = result.pass_at_1
        print(f"  [substrate] {bench}: {result.pass_at_1:.2%}")

    print("\n=== Gate 2: Rune (substrate + hypernetwork adapter) ===")
    from rune_runner import run_hypernetwork

    tmp_dir = tempfile.mkdtemp(prefix="rune_gate2_")

    def adapter_generator(prompt: str) -> str | None:
        return run_hypernetwork(
            trajectory_text=prompt,
            output_dir=tmp_dir,
            base_model_id=args.model,
            checkpoint_path=args.hypernet_checkpoint,
            device=args.device,
        )

    rune_stack = AdapterStack(
        base_model=args.model,
        adapter_ids=[args.warm_start_adapter],
        adapter_paths={},
        provider=provider,
        adapter_generator=adapter_generator,
    )
    rune_scores: dict[str, float] = {}
    for bench in benchmarks:
        result = run_benchmark(rune_stack, bench)
        rune_scores[bench] = result.pass_at_1
        print(f"  [rune] {bench}: {result.pass_at_1:.2%}")

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
        "hypernet_checkpoint": args.hypernet_checkpoint,
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
