"""Orchestrate benchmark evaluation: generate completions, score, compare.

Runs both base and LoRA variants for a given tier, invokes EvalPlus
scoring, and produces a comparison report.

Usage:
    uv run scripts/eval/run_benchmarks.py --tier 1
    uv run scripts/eval/run_benchmarks.py --tier 2 --model google/gemma-2-2b-it
    uv run scripts/eval/run_benchmarks.py --tier 3 --backend vllm --model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config import Backend, BenchmarkName, EvalConfig, Tier


def score_evalplus(
    samples_path: Path,
    dataset: str,
) -> dict[str, Any]:
    """Run EvalPlus evaluation and return results.

    Args:
        samples_path: Path to samples.jsonl file.
        dataset: "humaneval" or "mbpp".

    Returns:
        Dict with pass@k scores and per-task results.
    """
    cmd = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--dataset",
        dataset,
        "--samples",
        str(samples_path),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  EvalPlus failed:\n{result.stderr}")
        return {"error": result.stderr}

    # EvalPlus writes results next to the samples file
    results_dir = samples_path.parent / (samples_path.stem + "_eval_results.json")
    # Try alternative naming patterns
    possible_results = [
        results_dir,
        samples_path.with_suffix(".json"),
        samples_path.parent / "eval_results.json",
    ]

    for rpath in possible_results:
        if rpath.exists():
            with open(rpath) as f:
                return json.load(f)

    # Parse from stdout as fallback
    print(f"  EvalPlus output:\n{result.stdout}")
    return {"stdout": result.stdout, "stderr": result.stderr}


def run_evaluation(
    config: EvalConfig,
    mode: str,
    adapter_path: str | None,
    timestamp: str,
) -> dict[str, Any]:
    """Run completion generation and scoring for one mode (base or lora).

    Returns dict of benchmark results.
    """
    from generate_completions import run as generate_run

    eval_config = EvalConfig(
        model_id=config.model_id,
        adapter_path=adapter_path,
        backend=config.backend,
        vllm_base_url=config.vllm_base_url,
        tier=config.tier,
        output_dir=config.output_dir,
        seed=config.seed,
    )

    print(f"\n{'=' * 60}")
    print(f"  Generating completions: mode={mode}")
    print(f"{'=' * 60}")

    output_files = generate_run(eval_config)

    results: dict[str, Any] = {}
    for key, samples_path in output_files.items():
        benchmark_name = key.rsplit("_pass@", 1)[0]

        if benchmark_name in (
            BenchmarkName.HUMANEVAL.value,
            BenchmarkName.MBPP.value,
        ):
            print(f"\n--- Scoring {key} ---")
            eval_result = score_evalplus(samples_path, benchmark_name)
            results[key] = eval_result
        else:
            results[key] = {
                "note": f"Scoring for {benchmark_name} requires bigcodebench CLI"
            }

    return results


def run_orchestrator(config: EvalConfig) -> Path:
    """Run the full evaluation pipeline for both base and LoRA modes.

    Returns path to the results directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.output_dir) / timestamp

    all_results: dict[str, Any] = {
        "config": {
            "model_id": config.model_id,
            "adapter_path": config.adapter_path,
            "backend": config.backend.value,
            "tier": config.tier.value,
            "seed": config.seed,
        },
        "base": {},
        "lora": {},
    }

    # Run base model
    all_results["base"] = run_evaluation(config, "base", None, timestamp)

    # Run LoRA model if adapter is configured
    if config.adapter_path:
        all_results["lora"] = run_evaluation(
            config, "lora", config.adapter_path, timestamp
        )
    else:
        print("\nNo adapter path configured, skipping LoRA run.")

    # Save raw results
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate comparison report
    if config.adapter_path:
        from compare_results import generate_report

        report_path = results_dir / "report.md"
        generate_report(all_results, report_path)
        print(f"Report saved to {report_path}")

    return results_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run coding benchmark evaluation pipeline"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Evaluation tier (1=smoke, 2=mini, 3=full)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to PEFT adapter (enables LoRA comparison)",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default=None,
        help="Inference backend",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    config = EvalConfig(
        model_id=args.model or EvalConfig().model_id,
        adapter_path=args.adapter_path or EvalConfig().adapter_path,
        backend=Backend(args.backend) if args.backend else EvalConfig().backend,
        tier=Tier(args.tier),
        output_dir=args.output_dir,
    )

    print(f"Model: {config.model_id}")
    print(f"Backend: {config.backend.value}")
    print(f"Tier: {config.tier.value}")
    if config.adapter_path:
        print(f"Adapter: {config.adapter_path}")
    else:
        print("Adapter: None (base model only)")

    results_dir = run_orchestrator(config)
    print(f"\nAll results in: {results_dir}")


if __name__ == "__main__":
    main()
