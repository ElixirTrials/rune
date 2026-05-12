"""Gate 3: Procedural-encoding strength (§4.3).

Evaluates 15 OOD functions × 8 held-out inputs via exact-match output
comparison. Compares substrate baseline vs Rune (substrate + hypernetwork
adapter). Applies paired McNemar + Bonferroni.

Usage:
    uv run python scripts/paper/run_gate3.py \
        --hypernet-checkpoint path/to/checkpoint.bin \
        --output evaluation_results/gate3.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

from scripts.paper.statistical_tests import (
    bonferroni_correct,
    mcnemar_test,
    wilson_score_ci,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def _generate_completions(
    tasks: list[dict[str, str]],
    model: str,
    provider: Any,
    adapter_generator: Any | None = None,
) -> dict[str, str]:
    """Generate completions for all OOD tasks.

    Args:
        tasks: OOD task dicts with "task_id" and "prompt" fields.
        model: Base model ID.
        provider: InferenceProvider instance.
        adapter_generator: Optional callable(prompt) -> adapter_path for Rune.

    Returns:
        Dict mapping task_id -> generated completion.
    """
    loop = asyncio.new_event_loop()
    completions: dict[str, str] = {}
    try:
        for task in tasks:
            task_id = task["task_id"]
            prompt = task["prompt"]

            adapter_id_to_unload: str | None = None
            try:
                if adapter_generator is not None:
                    adapter_path = adapter_generator(prompt)
                    if adapter_path is not None:
                        adapter_id_to_unload = f"hypernet_{task_id}"
                        loop.run_until_complete(
                            provider.load_adapter(adapter_id_to_unload, adapter_path)
                        )

                gen_adapter_id = adapter_id_to_unload
                result = loop.run_until_complete(
                    provider.generate(
                        prompt=prompt,
                        model=model,
                        adapter_id=gen_adapter_id,
                        max_tokens=512,
                    )
                )
                completions[task_id] = result.text
            finally:
                if adapter_id_to_unload is not None:
                    try:
                        loop.run_until_complete(
                            provider.unload_adapter(adapter_id_to_unload)
                        )
                    except Exception:
                        logger.debug(
                            "Failed to unload adapter %s", adapter_id_to_unload
                        )
    finally:
        loop.close()

    return completions


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3: OOD procedural encoding")
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
    parser.add_argument("--n-inputs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/gate3.json")
    )
    args = parser.parse_args()

    import subprocess

    from evaluation.ood_benchmark import run_ood_benchmark
    from inference.factory import get_provider
    from model_training.training_common import (
        mlflow_log_artifact,
        mlflow_log_params,
        setup_mlflow,
    )
    from rune_runner import run_hypernetwork  # type: ignore[import-not-found]

    mlflow_ok = setup_mlflow("paper-gate3", tracking_uri=None)
    if mlflow_ok:
        import mlflow

        mlflow.start_run(run_name="gate3")
        mlflow_log_params(
            {
                "model": args.model,
                "warm_start_adapter": args.warm_start_adapter,
                "hypernet_checkpoint": args.hypernet_checkpoint,
                "n_inputs": args.n_inputs,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
            }
        )

    ood_data_path = Path("libs/evaluation/src/evaluation/data/ood_tasks.json")
    with ood_data_path.open() as f:
        all_tasks: list[dict[str, str]] = json.load(f)

    print(f"Gate 3: {len(all_tasks)} OOD tasks × {args.n_inputs} inputs")
    print(f"Model: {args.model}")
    print(f"Substrate: {args.warm_start_adapter}")
    print(f"Hypernetwork: {args.hypernet_checkpoint}")

    provider = get_provider()

    print("\n=== Substrate baseline ===")
    substrate_completions = _generate_completions(
        all_tasks,
        args.model,
        provider,
    )
    substrate_result = run_ood_benchmark(
        adapter_id=None,
        completions=substrate_completions,
        benchmark_name="ood_python_substrate",
    )
    print(
        f"  Pass rate: {substrate_result['ood_pass_rate']:.2%} "
        f"({substrate_result['pass_count']}/{substrate_result['total']})"
    )

    print("\n=== Rune (substrate + hypernetwork adapter) ===")
    tmp_dir = tempfile.mkdtemp(prefix="rune_gate3_")

    def adapter_generator(prompt: str) -> str | None:
        return run_hypernetwork(
            trajectory_text=prompt,
            output_dir=tmp_dir,
            base_model_id=args.model,
            checkpoint_path=args.hypernet_checkpoint,
            device=args.device,
        )

    rune_completions = _generate_completions(
        all_tasks,
        args.model,
        provider,
        adapter_generator=adapter_generator,
    )
    rune_result = run_ood_benchmark(
        adapter_id="rune_hypernet",
        completions=rune_completions,
        benchmark_name="ood_python_rune",
    )
    print(
        f"  Pass rate: {rune_result['ood_pass_rate']:.2%} "
        f"({rune_result['pass_count']}/{rune_result['total']})"
    )

    substrate_by_task = {
        r["task_id"]: r["passed"] for r in substrate_result["task_results"]
    }
    rune_by_task = {r["task_id"]: r["passed"] for r in rune_result["task_results"]}

    common_ids = sorted(set(substrate_by_task) & set(rune_by_task))
    paired = [(substrate_by_task[tid], rune_by_task[tid]) for tid in common_ids]
    mcnemar = mcnemar_test(paired)
    bonferroni = bonferroni_correct([mcnemar["p_value"]])

    n_sub = sum(1 for tid in common_ids if substrate_by_task[tid])
    n_rune = sum(1 for tid in common_ids if rune_by_task[tid])
    ci_substrate = wilson_score_ci(len(common_ids), n_sub)
    ci_rune = wilson_score_ci(len(common_ids), n_rune)

    print("\n=== Statistical Tests ===")
    print(f"  McNemar chi2={mcnemar['chi2']:.3f}, p={mcnemar['p_value']:.4f}")
    print(f"  Substrate CI: [{ci_substrate[0]:.3f}, {ci_substrate[1]:.3f}]")
    print(f"  Rune CI: [{ci_rune[0]:.3f}, {ci_rune[1]:.3f}]")

    report: dict[str, Any] = {
        "n_tasks": len(all_tasks),
        "n_inputs_per_task": args.n_inputs,
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "hypernet_checkpoint": args.hypernet_checkpoint,
        "substrate": {
            "pass_rate": substrate_result["ood_pass_rate"],
            "pass_count": substrate_result["pass_count"],
            "total": substrate_result["total"],
            "ci_95": {"lower": ci_substrate[0], "upper": ci_substrate[1]},
            "task_results": substrate_result["task_results"],
        },
        "rune": {
            "pass_rate": rune_result["ood_pass_rate"],
            "pass_count": rune_result["pass_count"],
            "total": rune_result["total"],
            "ci_95": {"lower": ci_rune[0], "upper": ci_rune[1]},
            "task_results": rune_result["task_results"],
        },
        "mcnemar": mcnemar,
        "bonferroni": bonferroni,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    delta = rune_result["ood_pass_rate"] - substrate_result["ood_pass_rate"]
    print(f"\nDelta (Rune - Substrate): {delta:+.2%}")
    print(f"Output: {args.output}")

    if mlflow_ok:
        mlflow.log_metric("substrate_ood_pass_rate", substrate_result["ood_pass_rate"])
        mlflow.log_metric("rune_ood_pass_rate", rune_result["ood_pass_rate"])
        mlflow.log_metric("ood_delta", delta)
        mlflow.log_metric("mcnemar_p_value", mcnemar["p_value"])
        mlflow_log_artifact(str(args.output))
        mlflow.end_run()


if __name__ == "__main__":
    main()
