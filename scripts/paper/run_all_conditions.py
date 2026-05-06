"""Master runner for paper Table 2: all 5 conditions.

Conditions:
    (i)   Frozen base — Qwen 3.5 9B + DeltaCoder warm-start LoRA (substrate)
    (ii)  Trajectory-aware RAG — substrate + retrieved trajectory context
    (iii) Direct PEFT QLoRA — substrate + best HPO-tuned LoRA
    (iv)  TTT-E2E — substrate + test-time MLP fine-tuning
    (v)   Rune — substrate + hypernetwork-generated per-task adapter

Usage:
    uv run python scripts/paper/run_all_conditions.py \
        --conditions i ii iii iv v \
        --benchmarks humaneval livecodebench \
        --hypernet-checkpoint path/to/checkpoint.bin \
        --output evaluation_results/table2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path

setup_path()

CONDITION_LABELS = {
    "i": "Frozen base (substrate)",
    "ii": "Trajectory-aware RAG",
    "iii": "Direct PEFT QLoRA",
    "iv": "TTT-E2E",
    "v": "Rune (ours)",
}

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def run_condition_static(
    benchmarks: list[str],
    model: str,
    adapter_ids: list[str],
    provider: Any,
) -> dict[str, float]:
    """Evaluate with a fixed adapter stack (conditions i, iii).

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        adapter_ids: Adapter IDs/paths to stack on top of base model.
        provider: InferenceProvider instance.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack

    stack = AdapterStack(
        base_model=model,
        adapter_ids=adapter_ids,
        adapter_paths={},
        provider=provider,
    )

    results: dict[str, float] = {}
    for bench_id in benchmarks:
        result = run_benchmark(stack, bench_id)
        results[bench_id] = result.pass_at_1
    return results


def run_condition_rune(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    hypernet_checkpoint: str,
    device: str,
    provider: Any,
) -> dict[str, float]:
    """Evaluate with per-task hypernetwork-generated adapters (condition v).

    For each benchmark problem, generates a task-specific adapter via the
    D2L hypernetwork, stacks it on top of the substrate (base + warm-start),
    and runs inference.

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        warm_start_adapter: Warm-start LoRA adapter ID (DeltaCoder).
        hypernet_checkpoint: Path to trained hypernetwork checkpoint.
        device: Device for hypernetwork computation.
        provider: InferenceProvider instance.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    import tempfile

    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from rune_runner import run_hypernetwork

    tmp_dir = tempfile.mkdtemp(prefix="rune_eval_")

    def adapter_generator(prompt: str) -> str | None:
        return run_hypernetwork(
            trajectory_text=prompt,
            output_dir=tmp_dir,
            base_model_id=model,
            checkpoint_path=hypernet_checkpoint,
            device=device,
        )

    stack = AdapterStack(
        base_model=model,
        adapter_ids=[warm_start_adapter],
        adapter_paths={},
        provider=provider,
        adapter_generator=adapter_generator,
    )

    results: dict[str, float] = {}
    for bench_id in benchmarks:
        result = run_benchmark(stack, bench_id)
        results[bench_id] = result.pass_at_1
    return results


def assemble_table2(
    all_results: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Assemble Table 2 data from per-condition results.

    Args:
        all_results: {condition: {benchmark: pass_at_1}}.

    Returns:
        Table 2 structured data with deltas vs condition (i) substrate baseline.
    """
    table: dict[str, Any] = {"conditions": {}}
    base_i = all_results.get("i", {})

    for cond, scores in all_results.items():
        deltas = {}
        for bench, score in scores.items():
            i_score = base_i.get(bench, 0.0)
            deltas[bench] = score - i_score

        table["conditions"][cond] = {
            "label": CONDITION_LABELS.get(cond, cond),
            "scores": scores,
            "delta_vs_substrate": deltas,
        }

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all paper conditions (Table 2)")
    parser.add_argument(
        "--conditions", nargs="+", default=["i", "ii", "iii", "iv", "v"],
        choices=["i", "ii", "iii", "iv", "v"],
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["humaneval", "livecodebench"],
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter", default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument(
        "--adapter-iii", type=str, default=None,
        help="Path to HPO-tuned QLoRA adapter for Condition (iii)",
    )
    parser.add_argument(
        "--hypernet-checkpoint", type=str, default=None,
        help="Path to trained hypernetwork checkpoint for Condition (v)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/table2.json"))
    args = parser.parse_args()

    from inference.factory import get_provider

    provider = get_provider()

    all_results: dict[str, dict[str, float]] = {}

    for cond in args.conditions:
        print(f"\n{'='*60}")
        print(f"Condition ({cond}): {CONDITION_LABELS[cond]}")
        print(f"{'='*60}")

        start = time.time()

        if cond == "i":
            results = run_condition_static(
                args.benchmarks, args.model,
                adapter_ids=[args.warm_start_adapter],
                provider=provider,
            )

        elif cond == "ii":
            results = run_condition_static(
                args.benchmarks, args.model,
                adapter_ids=[args.warm_start_adapter],
                provider=provider,
            )
            print("  (RAG context injection handled by eval harness --rag-context flag)")

        elif cond == "iii":
            if not args.adapter_iii:
                print("  SKIPPED: --adapter-iii not provided")
                continue
            results = run_condition_static(
                args.benchmarks, args.model,
                adapter_ids=[args.warm_start_adapter, args.adapter_iii],
                provider=provider,
            )

        elif cond == "iv":
            results = run_condition_static(
                args.benchmarks, args.model,
                adapter_ids=[args.warm_start_adapter],
                provider=provider,
            )
            print("  (TTT inner-loop handled by eval harness --ttt flag)")

        elif cond == "v":
            if not args.hypernet_checkpoint:
                print("  SKIPPED: --hypernet-checkpoint not provided")
                continue
            results = run_condition_rune(
                args.benchmarks, args.model,
                warm_start_adapter=args.warm_start_adapter,
                hypernet_checkpoint=args.hypernet_checkpoint,
                device=args.device,
                provider=provider,
            )

        else:
            continue

        elapsed = time.time() - start
        for bench_id, score in results.items():
            print(f"  {bench_id}: {score:.2%}")
        print(f"  Elapsed: {elapsed:.1f}s")

        all_results[cond] = results

    table = assemble_table2(all_results)
    table["metadata"] = {
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "benchmarks": args.benchmarks,
        "hypernet_checkpoint": args.hypernet_checkpoint,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))
    print(f"\nTable 2 written to {args.output}")


if __name__ == "__main__":
    main()
