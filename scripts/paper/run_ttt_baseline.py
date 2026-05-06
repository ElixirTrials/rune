"""Run Condition (iv) TTT-E2E baseline evaluation.

Usage:
    uv run python scripts/paper/run_ttt_baseline.py \
        --model Qwen/Qwen3.5-9B \
        --lr 1e-4 \
        --output evaluation_results/condition_iv.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Condition (iv): TTT-E2E baseline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mlp-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/condition_iv.json"))
    args = parser.parse_args()

    from model_training.ttt_e2e import TTTConfig

    config = TTTConfig(
        mlp_fraction=args.mlp_fraction,
        inner_lr=args.lr,
        inner_steps=args.steps,
    )

    print(f"TTT-E2E config: {config}")
    print("Run the full eval via the benchmark harness with --ttt flag.")

    result = {
        "condition": "iv_ttt_e2e",
        "model": args.model,
        "config": {
            "mlp_fraction": config.mlp_fraction,
            "inner_lr": config.inner_lr,
            "inner_steps": config.inner_steps,
        },
        "status": "ready_for_eval",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
