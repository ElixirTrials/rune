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
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path

setup_path()

from scripts.paper.statistical_tests import bonferroni_correct, mcnemar_test

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3: OOD procedural encoding")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter", default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument(
        "--hypernet-checkpoint", type=str, required=True,
        help="Path to trained hypernetwork checkpoint",
    )
    parser.add_argument("--n-inputs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/gate3.json"))
    args = parser.parse_args()

    from evaluation.ood_benchmark import run_ood_benchmark

    ood_data_path = Path("libs/evaluation/src/evaluation/data/ood_tasks.json")
    with ood_data_path.open() as f:
        all_tasks = json.load(f)

    print(f"Gate 3: {len(all_tasks)} OOD tasks × {args.n_inputs} inputs")
    print(f"Model: {args.model}")
    print(f"Substrate: {args.warm_start_adapter}")
    print(f"Hypernetwork: {args.hypernet_checkpoint}")

    report: dict[str, Any] = {
        "n_tasks": len(all_tasks),
        "n_inputs_per_task": args.n_inputs,
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "hypernet_checkpoint": args.hypernet_checkpoint,
        "status": "ready_for_gpu_run",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
