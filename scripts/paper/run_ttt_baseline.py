"""Run Condition (iv) TTT-E2E baseline evaluation.

Loads the base model, applies test-time training (inner-loop MLP fine-tuning)
on each benchmark problem's prompt, then generates the completion.

Usage:
    uv run python scripts/paper/run_ttt_baseline.py \
        --model Qwen/Qwen3.5-9B \
        --benchmarks humaneval livecodebench \
        --output evaluation_results/condition_iv.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Condition (iv): TTT-E2E baseline")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter",
        default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mlp-fraction", type=float, default=0.25)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "livecodebench"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/condition_iv.json")
    )
    args = parser.parse_args()

    import torch
    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from inference.factory import get_provider
    from model_training.ttt_e2e import TTTConfig, ttt_forward_pass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ttt_config = TTTConfig(
        mlp_fraction=args.mlp_fraction,
        inner_lr=args.lr,
        inner_steps=args.steps,
    )
    print(f"TTT-E2E config: {ttt_config}")

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ttt_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    original_sd = {k: v.cpu().clone() for k, v in ttt_model.state_dict().items()}

    def completion_override(prompt: str, max_tokens: int) -> str:
        ttt_model.load_state_dict(original_sd)
        result = ttt_forward_pass(
            model=ttt_model,
            tokenizer=tokenizer,
            context=prompt,
            query=prompt,
            config=ttt_config,
        )
        return result["generation"]

    provider = get_provider()
    stack = AdapterStack(
        base_model=args.model,
        adapter_ids=[args.warm_start_adapter],
        adapter_paths={},
        provider=provider,
        completion_override=completion_override,
    )

    all_results: dict[str, float] = {}
    for bench_id in args.benchmarks:
        print(f"\nEvaluating {bench_id}...")
        start = time.time()
        result = run_benchmark(stack, bench_id)
        elapsed = time.time() - start
        all_results[bench_id] = result.pass_at_1
        print(f"  {bench_id}: {result.pass_at_1:.2%} ({elapsed:.1f}s)")

    output = {
        "condition": "iv_ttt_e2e",
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "config": {
            "mlp_fraction": ttt_config.mlp_fraction,
            "inner_lr": ttt_config.inner_lr,
            "inner_steps": ttt_config.inner_steps,
        },
        "benchmarks": all_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
