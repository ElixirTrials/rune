"""CLI entrypoint for round-2 (oracle-teacher) hypernetwork training.

Usage:
    uv run scripts/train_round2.py \\
        --sakana-checkpoint-path /path/to/sakana.bin \\
        --oracle-registry-url sqlite:///~/.rune/adapters.db \\
        --dataset-path data/phase_corpus/all_bins_concat.jsonl \\
        --num-steps 1000 \\
        --checkpoint-dir ./checkpoints/round2 \\
        --experiment-name d2l-qwen3-round2 \\
        --max-loaded-oracles 4 \\
        --min-oracle-coverage 0.8 \\
        --oracle-fallback skip
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from model_training.round2_config import Round2TrainConfig
from model_training.round2_train import train_d2l_qwen3_round2


def build_config(argv: Sequence[str]) -> Round2TrainConfig:
    """Parse argv into a :class:`Round2TrainConfig`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sakana-checkpoint-path", required=True)
    parser.add_argument("--oracle-registry-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--full-checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/round2")
    parser.add_argument("--experiment-name", default="d2l-qwen3-round2")
    parser.add_argument("--max-loaded-oracles", type=int, default=4)
    parser.add_argument("--min-oracle-coverage", type=float, default=0.8)
    parser.add_argument(
        "--oracle-fallback", choices=["base_model", "skip"], default="skip"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--kill-switch-enabled", action="store_true")
    parser.add_argument("--kill-switch-step-cadence", type=int, default=100)
    parser.add_argument("--kill-switch-benchmark-id", default="humaneval")
    parser.add_argument("--kill-switch-max-samples", type=int, default=10)
    parser.add_argument("--kill-switch-delta", type=float, default=0.05)

    ns = parser.parse_args(argv)
    return Round2TrainConfig(
        sakana_checkpoint_path=ns.sakana_checkpoint_path,
        oracle_registry_url=ns.oracle_registry_url,
        dataset_path=ns.dataset_path,
        num_steps=ns.num_steps,
        lr=ns.lr,
        alpha=ns.alpha,
        temperature=ns.temperature,
        checkpoint_every=ns.checkpoint_every,
        full_checkpoint_every=ns.full_checkpoint_every,
        checkpoint_dir=ns.checkpoint_dir,
        experiment_name=ns.experiment_name,
        max_loaded_oracles=ns.max_loaded_oracles,
        min_oracle_coverage=ns.min_oracle_coverage,
        oracle_fallback=ns.oracle_fallback,
        dry_run=ns.dry_run,
        smoke_test=ns.smoke_test,
        max_length=ns.max_length,
        grad_clip=ns.grad_clip,
        warmup_steps=ns.warmup_steps,
        kill_switch_enabled=ns.kill_switch_enabled,
        kill_switch_step_cadence=ns.kill_switch_step_cadence,
        kill_switch_benchmark_id=ns.kill_switch_benchmark_id,
        kill_switch_max_samples=ns.kill_switch_max_samples,
        kill_switch_delta=ns.kill_switch_delta,
    )


def main(argv: Sequence[str]) -> int:
    """Run round-2 training from the command line."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = build_config(argv)
    report = train_d2l_qwen3_round2(config)
    logging.info("Round-2 run report: %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
