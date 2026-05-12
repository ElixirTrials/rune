"""Launch HyperLoRA training on SageMaker.

Wraps ``scripts/run_hypernet.sh`` as a SageMaker PyTorch Estimator with:
- Spot instance support (up to 70% savings)
- Instance fleet mode (multiple GPU types, priority-ordered fallback)
- S3-backed checkpointing for spot interruption recovery
- MLflow tracking URI passthrough

Usage:
    # Single instance type:
    uv run python scripts/launch_sagemaker.py \
        --instance-type ml.g5.2xlarge

    # Fleet mode (spot-friendly, auto-fallback):
    uv run python scripts/launch_sagemaker.py --fleet

    # Dry run (print config, don't launch):
    uv run python scripts/launch_sagemaker.py --fleet --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path  # type: ignore[import-not-found]

setup_path()

logger = logging.getLogger(__name__)

_DEFAULT_ROLE_ARN = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    "arn:aws:iam::role/SageMakerTrainingRole",
)
_DEFAULT_BUCKET = os.environ.get(
    "SAGEMAKER_BUCKET",
    "elixirtrials-949678234935-eu-west-2-artifacts",
)
_DEFAULT_S3_PREFIX = "rune/hypernet-training"

_FLEET_INSTANCES = [
    {"InstanceType": "ml.g5.2xlarge", "WeightedCapacity": 1},
    {"InstanceType": "ml.g5.4xlarge", "WeightedCapacity": 1},
    {"InstanceType": "ml.g6.2xlarge", "WeightedCapacity": 1},
    {"InstanceType": "ml.p3.2xlarge", "WeightedCapacity": 1},
]

_VRAM_TIER_MAP = {
    "ml.g5.2xlarge": "low",
    "ml.g5.4xlarge": "low",
    "ml.g6.2xlarge": "low",
    "ml.p3.2xlarge": "low",
    "ml.g5.12xlarge": "mid",
    "ml.g5.48xlarge": "mid",
    "ml.p3.8xlarge": "mid",
    "ml.p4d.24xlarge": "high",
    "ml.p5.48xlarge": "high",
}


def _build_hyperparameters(args: argparse.Namespace) -> dict[str, str]:
    hp: dict[str, str] = {
        "num-steps": str(args.num_steps),
        "experiment-name": args.experiment_name,
        "base-model": args.base_model,
    }
    if args.vram_tier:
        hp["vram-tier"] = args.vram_tier
    if args.mlflow_tracking_uri:
        hp["mlflow-tracking-uri"] = args.mlflow_tracking_uri
    if args.smoke:
        hp["smoke"] = "1"
    return hp


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HyperLoRA training on SageMaker")

    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.g5.2xlarge",
        help="SageMaker instance type (ignored in fleet mode).",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fleet",
        action="store_true",
        help="Use instance fleet mode with spot fallback across GPU types.",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        default=True,
        help="Use managed spot instances (default: True).",
    )
    parser.add_argument("--no-spot", action="store_false", dest="spot")
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=48,
        help="Max wait time for spot capacity (hours).",
    )
    parser.add_argument(
        "--max-run-hours",
        type=float,
        default=24,
        help="Max training job runtime (hours).",
    )

    parser.add_argument("--role-arn", type=str, default=_DEFAULT_ROLE_ARN)
    parser.add_argument("--bucket", type=str, default=_DEFAULT_BUCKET)
    parser.add_argument("--s3-prefix", type=str, default=_DEFAULT_S3_PREFIX)

    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--experiment-name", type=str, default="hypernet-hpo-sm")
    parser.add_argument("--vram-tier", type=str, default=None, choices=["low", "mid", "high"])
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without launching.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import sagemaker  # noqa: PLC0415
    from sagemaker.pytorch import PyTorch  # noqa: PLC0415

    session = sagemaker.Session()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"hypernet-hpo-{timestamp}"

    s3_base = f"s3://{args.bucket}/{args.s3_prefix}"
    checkpoint_s3 = f"{s3_base}/checkpoints/{job_name}"
    output_s3 = f"{s3_base}/output/{job_name}"

    hyperparameters = _build_hyperparameters(args)

    estimator_kwargs: dict = {
        "entry_point": "run_hypernet.sh",
        "source_dir": "scripts",
        "role": args.role_arn,
        "instance_count": args.instance_count,
        "framework_version": "2.11.0",
        "py_version": "py312",
        "hyperparameters": hyperparameters,
        "output_path": output_s3,
        "checkpoint_s3_uri": checkpoint_s3,
        "checkpoint_local_path": "/opt/ml/checkpoints",
        "max_run": int(args.max_run_hours * 3600),
        "sagemaker_session": session,
        "environment": {
            "SM_HP_CHECKPOINT_DIR": "/opt/ml/checkpoints",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "disable_output_compression": True,
    }

    if args.spot:
        estimator_kwargs["use_spot_instances"] = True
        estimator_kwargs["max_wait"] = int(args.max_wait_hours * 3600)

    if args.fleet:
        estimator_kwargs["instance_groups"] = [
            sagemaker.InstanceGroup("gpu", args.instance_type, args.instance_count)
        ]
        logger.info("Fleet mode: %s", json.dumps(_FLEET_INSTANCES, indent=2))
    else:
        estimator_kwargs["instance_type"] = args.instance_type

    estimator = PyTorch(**estimator_kwargs)

    logger.info("Job name:       %s", job_name)
    logger.info("Instance:       %s (fleet=%s, spot=%s)", args.instance_type, args.fleet, args.spot)
    logger.info("Checkpoints:    %s", checkpoint_s3)
    logger.info("Output:         %s", output_s3)
    logger.info("Hyperparams:    %s", json.dumps(hyperparameters, indent=2))

    if args.dry_run:
        logger.info("Dry run — not launching. Config above is what would be used.")
        return

    estimator.fit(job_name=job_name, wait=False)
    logger.info("Job submitted: %s", job_name)
    logger.info("Monitor: aws sagemaker describe-training-job --training-job-name %s", job_name)


if __name__ == "__main__":
    main()
