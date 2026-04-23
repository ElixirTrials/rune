"""Phase Corpus Producer — self-distillation oracle corpus for phase-aware training.

For each (benchmark, problem), runs the full 5-phase Rune pipeline, filters
by Pass@1=1.0, bins per-phase artifacts into 25 oracle bins, emits JSONL
manifests, and invokes QLoRA training per bin.

Usage:
    uv run scripts/phase_corpus_producer.py \\
        --benchmark humaneval \\
        --out-dir data/phase_corpus \\
        --max-problems 20

    uv run scripts/phase_corpus_producer.py \\
        --benchmark humaneval mbpp apps \\
        --out-dir data/phase_corpus \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path  # type: ignore[import]

setup_path()

from corpus_producer.manifest import emit_bin_manifest
from corpus_producer.models import PhaseArtifact
from corpus_producer.pipeline_runner import run_pipeline_for_problem
from corpus_producer.progress_db import ProgressDB
from corpus_producer.rationalization import MIN_EXAMPLES_PER_BIN, star_rationalize
from corpus_producer.s3_uploader import upload_manifest
from corpus_producer.success_filter import filter_artifacts
from corpus_producer.trainer_bridge import invoke_bin_training

logger = logging.getLogger(__name__)

BENCHMARKS = [
    "humaneval",
    "mbpp",
    "apps",
    "bigcodebench",
    "ds_1000",
    "livecodebench",
]
PIPELINE_TIMEOUT_DEFAULT = 300


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase_corpus_producer",
        description=(
            "Self-distillation oracle corpus producer for Rune phase training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
        metavar="BENCHMARK",
        help="One or more benchmark ids to run.",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        metavar="PROBLEM_ID",
        default=None,
        help="Explicit problem ids to run (overrides --max-problems).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        metavar="N",
        help="Maximum problems per benchmark (None = all).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/phase_corpus",
        metavar="DIR",
        help="Output directory for JSONL manifests and progress DB.",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=int,
        default=PIPELINE_TIMEOUT_DEFAULT,
        dest="pipeline_timeout",
        metavar="SECS",
        help="Per-problem pipeline subprocess timeout (seconds).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if progress DB marks a problem as done.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        dest="skip_training",
        help="Emit manifests but do not invoke train_and_register.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run the pipeline and emit manifests; pass dry_run=True to trainer.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-9B",
        dest="base_model",
        metavar="MODEL_ID",
        help="Base model HF repo id for pipeline runs.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        dest="database_url",
        metavar="URL",
        help="SQLAlchemy URL for AdapterRegistry (defaults to env/default).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="rune-qlora",
        dest="mlflow_experiment",
        metavar="NAME",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        dest="s3_bucket",
        metavar="BUCKET",
        help=(
            "Optional S3 bucket to mirror bin manifests to. When unset, "
            "manifests remain local-only."
        ),
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        dest="s3_prefix",
        metavar="PREFIX",
        help=(
            "Key prefix within --s3-bucket (e.g. 'oracles/run-1'). "
            "Ignored when --s3-bucket is unset."
        ),
    )
    parser.add_argument(
        "--shard",
        default=None,
        dest="shard",
        metavar="IDX/TOTAL",
        help=(
            "Round-robin shard spec, e.g. '0/4' for shard 0 of 4 workers. "
            "Progress DB is shared; restarts are safe across shards."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        dest="cuda_visible_devices",
        metavar="DEVICES",
        help=(
            "Optional CUDA_VISIBLE_DEVICES value (e.g. '0') forwarded to each "
            "pipeline subprocess. Combine with --shard for per-GPU workers."
        ),
    )
    return parser


def _parse_shard(s: str) -> tuple[int, int]:
    """Parse a ``"<idx>/<total>"`` shard spec.

    Args:
        s: Shard spec, e.g. ``"0/4"`` for shard 0 of 4.

    Returns:
        Tuple of ``(idx, total)`` with ``0 <= idx < total`` and ``total > 0``.

    Raises:
        ValueError: On malformed spec, non-integer components, non-positive
            total, or out-of-range idx.
    """
    if "/" not in s:
        raise ValueError(f"--shard must be '<idx>/<total>', got {s!r}")
    idx_s, total_s = s.split("/", 1)
    try:
        idx = int(idx_s)
        total = int(total_s)
    except ValueError as exc:
        raise ValueError(f"--shard components must be integers, got {s!r}") from exc
    if total <= 0:
        raise ValueError(f"--shard total must be > 0, got {total}")
    if idx < 0 or idx >= total:
        raise ValueError(f"--shard idx must be in [0, {total - 1}], got {idx}")
    return idx, total


def apply_shard(
    problems: list[tuple[str, str]], idx: int, total: int
) -> list[tuple[str, str]]:
    """Round-robin slice a problem list for shard ``idx`` of ``total``.

    Round-robin (``problems[idx::total]``) balances per-problem runtime
    variance better than contiguous chunks. The union of all ``total``
    shards equals the input list with no overlaps.

    Args:
        problems: Full list of ``(problem_id, prompt)`` tuples.
        idx: Shard index (0-based, must be < total).
        total: Total number of shards.

    Returns:
        The subset of ``problems`` assigned to this shard.
    """
    return problems[idx::total]


def _load_problems(
    benchmark: str,
    problem_ids: list[str] | None,
    max_problems: int | None,
) -> list[tuple[str, str]]:
    """Return list of (problem_id, prompt) pairs for the given benchmark.

    Imports benchmark dataset loader lazily. Falls back to a stub list when
    the evaluation package is not yet installed (CPU-only CI).

    Args:
        benchmark: Benchmark identifier.
        problem_ids: Explicit list of problem ids (overrides max_problems).
        max_problems: Cap on number of problems; None = all.

    Returns:
        List of (problem_id, prompt) tuples.
    """
    try:
        from evaluation.benchmarks import load_problems

        problems = load_problems(
            benchmark, problem_ids=problem_ids, max_samples=max_problems
        )
        return [(p.problem_id, p.prompt) for p in problems]
    except ImportError:
        logger.warning(
            "evaluation.benchmarks not available — using stub problem list for %s",
            benchmark,
        )
        n = max_problems or 1
        return [
            (f"{benchmark.upper()}/{i}", f"Stub problem {i} for {benchmark}.")
            for i in range(n)
        ]


def produce_corpus(
    benchmarks: list[str],
    out_dir: Path,
    *,
    problem_ids: list[str] | None = None,
    max_problems: int | None = None,
    pipeline_timeout: int = PIPELINE_TIMEOUT_DEFAULT,
    force: bool = False,
    skip_training: bool = False,
    dry_run: bool = False,
    base_model: str = "Qwen/Qwen3.5-9B",
    database_url: str | None = None,
    mlflow_experiment: str = "rune-qlora",
    s3_bucket: str | None = None,
    s3_prefix: str = "",
    shard_idx: int = 0,
    shard_total: int = 1,
    cuda_visible_devices: str | None = None,
) -> dict[str, int]:
    """Main orchestration loop for phase corpus production.

    For each (benchmark, problem):
      1. Skip if already done in progress DB (unless --force).
      2. Run 5-phase pipeline via subprocess.
      3. Filter artifacts by Pass@1.
      4. Accumulate into per-bin artifact lists.

    After all problems:
      5. For each bin with < MIN_EXAMPLES_PER_BIN, run STaR rationalization.
      6. Emit JSONL manifests per bin.
      7. Invoke QLoRA training per bin (unless --skip-training / --dry-run).

    Args:
        benchmarks: Benchmark ids to process.
        out_dir: Root output directory.
        problem_ids: Explicit problem ids (applied across all benchmarks).
        max_problems: Cap per benchmark.
        pipeline_timeout: Per-problem subprocess timeout (seconds).
        force: Ignore progress DB and re-run all problems.
        skip_training: Emit manifests but skip training.
        dry_run: Pass dry_run=True to trainer; still emits manifests.
        base_model: HF repo id for pipeline subprocess.
        database_url: AdapterRegistry DB URL.
        mlflow_experiment: MLflow experiment name.

    Returns:
        Dict mapping bin_key -> number of training records in that bin.
    """
    out_dir = Path(out_dir)
    db = ProgressDB(out_dir / "progress.db")

    # Accumulated artifacts keyed by bin
    bin_artifacts_map: dict[str, list[PhaseArtifact]] = {}
    # Track failing problems per benchmark for rationalization
    failing_problems: list[tuple[str, str, str]] = []

    for benchmark in benchmarks:
        problems = _load_problems(benchmark, problem_ids, max_problems)
        if shard_total > 1:
            problems = apply_shard(problems, shard_idx, shard_total)
            logger.info(
                "Shard %d/%d: processing %d problems for benchmark %s",
                shard_idx,
                shard_total,
                len(problems),
                benchmark,
            )
        else:
            logger.info(
                "Processing %d problems for benchmark %s",
                len(problems),
                benchmark,
            )

        for problem_id, prompt in problems:
            # Resume: skip if all phases are already done
            if not force and db.is_done(benchmark, problem_id, "integrate"):
                logger.debug("Skipping done problem %s/%s", benchmark, problem_id)
                continue

            db.mark_running(benchmark, problem_id, "pipeline")
            result = run_pipeline_for_problem(
                benchmark,
                problem_id,
                prompt,
                timeout=pipeline_timeout,
                base_model_id=base_model,
                cuda_visible_devices=cuda_visible_devices,
            )

            if not result.success:
                db.mark_failed(benchmark, problem_id, "pipeline")
                failing_problems.append((benchmark, problem_id, prompt))
                continue

            kept = filter_artifacts(
                result.artifacts, result.final_code, benchmark, problem_id
            )

            if kept:
                for art in kept:
                    key = art.bin_key()
                    bin_artifacts_map.setdefault(key, []).append(art)
                db.mark_done(benchmark, problem_id, "integrate")
            else:
                db.mark_failed(benchmark, problem_id, "integrate")
                failing_problems.append((benchmark, problem_id, prompt))

    # STaR rationalization for thin bins
    for bin_key, arts in list(bin_artifacts_map.items()):
        if len(arts) < MIN_EXAMPLES_PER_BIN:
            logger.info(
                "Bin %s has %d examples (< %d); attempting rationalization.",
                bin_key,
                len(arts),
                MIN_EXAMPLES_PER_BIN,
            )
            new_arts = star_rationalize(
                bin_key,
                existing_artifacts=arts,
                failing_problems=failing_problems,
                test_hints_by_problem={},
                pipeline_runner=lambda bm, pid, p, **kw: run_pipeline_for_problem(
                    bm, pid, p, **kw
                ),
                success_filter_fn=filter_artifacts,
                timeout=pipeline_timeout,
                base_model_id=base_model,
            )
            bin_artifacts_map[bin_key].extend(new_arts)

    # Emit manifests and train
    manifests_dir = out_dir / "manifests"
    bin_record_counts: dict[str, int] = {}

    for bin_key, arts in bin_artifacts_map.items():
        if not arts:
            continue
        if db.is_bin_done(bin_key) and not force:
            logger.info("Bin %s already trained; skipping.", bin_key)
            bin_record_counts[bin_key] = len(arts)
            continue

        manifest_path = emit_bin_manifest(bin_key, arts, manifests_dir)
        bin_record_counts[bin_key] = len(arts)

        if s3_bucket:
            upload_manifest(manifest_path, bucket=s3_bucket, prefix=s3_prefix)

        if skip_training:
            logger.info(
                "--skip-training: manifest written, skipping training for %s.",
                bin_key,
            )
            continue

        db.mark_bin_training(bin_key)
        adapter_id = invoke_bin_training(
            bin_key,
            manifest_path,
            dry_run=dry_run,
            database_url=database_url,
            mlflow_experiment=mlflow_experiment,
        )
        db.mark_bin_done(bin_key, adapter_id)
        logger.info("Bin %s -> adapter %s", bin_key, adapter_id)

    db.close()
    return bin_record_counts


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args()

    shard_idx, shard_total = (0, 1)
    if args.shard:
        shard_idx, shard_total = _parse_shard(args.shard)

    counts = produce_corpus(
        benchmarks=args.benchmark,
        out_dir=Path(args.out_dir),
        problem_ids=args.problems,
        max_problems=args.max_problems,
        pipeline_timeout=args.pipeline_timeout,
        force=args.force,
        skip_training=args.skip_training,
        dry_run=args.dry_run,
        base_model=args.base_model,
        database_url=args.database_url,
        mlflow_experiment=args.mlflow_experiment,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        shard_idx=shard_idx,
        shard_total=shard_total,
        cuda_visible_devices=args.cuda_visible_devices,
    )

    total = sum(counts.values())
    print(f"Done. {len(counts)} bins, {total} total training records.")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} records")


if __name__ == "__main__":
    main()
