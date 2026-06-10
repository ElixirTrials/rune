#!/usr/bin/env python
"""Thin entry wrapper: load a YAML into DistillConfig and run Stage-2 distillation.

Usage (under the RAM watchdog):
    tools/run_guarded.sh <log> tools/_distill_entry.py --config <yaml> [--max-steps N]

GPU run-and-observe path; no GPU imports here (they are deferred inside
run_hypernet_distillation). Lives under tools/ (not core src/rune), per CLAUDE.md.
"""

from __future__ import annotations

import argparse

import yaml

from rune.training.hypernet_distill import DistillConfig, run_hypernet_distillation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to the distill YAML")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override DistillConfig.max_steps (smoke runs)",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        data = yaml.safe_load(f) or {}

    # Fail loud on a stray/misspelled key (pydantic v2 silently drops extras).
    unknown = set(data) - set(DistillConfig.model_fields)
    if unknown:
        raise SystemExit(f"unknown DistillConfig fields: {sorted(unknown)}")

    cfg = DistillConfig(**data)
    if args.max_steps is not None:
        cfg = cfg.model_copy(update={"max_steps": args.max_steps})

    run_hypernet_distillation(cfg)


if __name__ == "__main__":
    main()
