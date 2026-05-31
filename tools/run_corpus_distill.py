"""Launch Stage-1 D2L corpus distillation (issue #49). Run under run_guarded.sh.

Builds a DistillConfig and calls run_hypernet_distillation, which logs online
metrics (loss, diff_agreement, preservation, diff_token_frac, scaler_B, grad norms)
to JSONL + MLflow (localhost:5000) and early-stops on collapse/no-learning.
"""
from __future__ import annotations

import argparse
import sys

from rune.training.hypernet_distill import DistillConfig, run_hypernet_distillation

S3_CKPT = ("s3://elixirtrials-949678234935-eu-west-2-artifacts/"
           "checkpoints/hypernet_hpo/checkpoint.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/tmp/rune-corpus/external_codereview.unrolled.jsonl")
    ap.add_argument("--checkpoint", default=S3_CKPT)
    ap.add_argument("--out", default="/tmp/rune-ck-issue49")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--scaler-b-init", type=float, default=0.1)
    ap.add_argument("--train-scaling", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--val-corpus", default="")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-steps", type=int, default=200)
    ap.add_argument("--exp", default="issue49-d2l-corpus")
    a = ap.parse_args()
    cfg = DistillConfig(
        corpus_path=a.corpus,
        checkpoint_path=a.checkpoint,
        checkpoint_dir=a.out,
        max_steps=a.max_steps,
        scaler_b_init=a.scaler_b_init,
        train_scaling=a.train_scaling,
        learning_rate=a.lr,
        num_epochs=a.epochs,
        max_seq_length=a.max_seq_length,
        val_corpus_path=a.val_corpus,
        grad_accum_steps=a.grad_accum,
        val_steps=a.val_steps,
        experiment_name=a.exp,
        save_steps=200,
        log_steps=10,
    )
    run_hypernet_distillation(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
