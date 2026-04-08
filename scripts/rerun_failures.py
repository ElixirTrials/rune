#!/usr/bin/env python3
"""Re-run only the failed problems from a previous benchmark against a new config.

Reads a previous benchmark result directory to extract which problems were
answered incorrectly, then runs just those problems using a fresh config
(typically one with the retriever enabled or a newer model snapshot).

Usage::

    uv run python scripts/rerun_failures.py \\
        --results-dir benchmark_results/20260325_072535_e15938d1_math_prompt/olym_math_easy \\
        --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml

    # Override output directory
    uv run python scripts/rerun_failures.py \\
        --results-dir benchmark_results/20260325_072535_e15938d1_math_prompt/olym_math_easy \\
        --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml \\
        --output-dir /tmp/rerun_out

    # Enable debug logging
    uv run python scripts/rerun_failures.py \\
        --results-dir benchmark_results/20260325_072535_e15938d1_math_prompt/olym_math_easy \\
        --config libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml \\
        --log-level DEBUG
"""
# ruff: noqa: T201
# mypy: ignore-errors
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "libs" / "evaluation" / "src"))
sys.path.insert(0, str(REPO_ROOT / "libs" / "shared" / "src"))
sys.path.insert(0, str(REPO_ROOT / "libs" / "inference" / "src"))


def _fresh_run_id(suffix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}_{suffix}"


def _find_results_file(results_dir: Path) -> Path:
    """Find results.json inside results_dir/{any_model_subdir}/results.json."""
    candidates = sorted(results_dir.glob("*/results.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No results.json found under {results_dir}. "
            "Expected: <results_dir>/<model_slug>/results.json"
        )
    if len(candidates) > 1:
        logging.getLogger(__name__).warning(
            "Multiple results.json found; using the first: %s", candidates[0]
        )
    return candidates[0]


def _load_failed_problems(results_file: Path) -> list[dict]:
    """Return the subset of result rows where correct=False."""
    with results_file.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {results_file}, got {type(data)}")

    failed = [r for r in data if not r.get("correct", True)]
    return failed


def _load_dataset_name(results_dir: Path) -> str:
    """Infer dataset name from summary.json inside the results directory."""
    candidates = sorted(results_dir.glob("*/summary.json"))
    if candidates:
        with candidates[0].open(encoding="utf-8") as fh:
            summary = json.load(fh)
        name = summary.get("dataset")
        if name:
            return name

    # Fall back to using the results_dir's own name as the dataset slug.
    return results_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run only the failed problems from a previous benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Path to the dataset-level benchmark results directory "
            "(e.g. benchmark_results/<run_id>/olym_math_easy). "
            "Must contain a <model_slug>/results.json file."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML benchmark config to use for the re-run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output_dir from the config.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the auto-generated run_id.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from evaluation.benchmark_runner import (
        _build_retriever,
        build_backend,
        run_dataset_benchmark,
    )
    from evaluation.config import load_config
    from evaluation.utils import (
        _dataset_slug,
        _model_slug,
        _resolve_system_prompt,
        compute_summary,
        save_results,
    )

    # ── 1. Find and parse the previous results ─────────────────────────────────
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        logger.error("--results-dir does not exist: %s", results_dir)
        sys.exit(1)

    results_file = _find_results_file(results_dir)
    logger.info("Reading previous results from %s", results_file)

    failed = _load_failed_problems(results_file)
    if not failed:
        print("\nNo failed problems found in the previous results — nothing to re-run.")
        sys.exit(0)

    failed_ids: set[str] = {r["problem_id"] for r in failed}
    logger.info(
        "Found %d failed problem(s): %s",
        len(failed),
        sorted(failed_ids),
    )

    # ── 2. Load the new config ─────────────────────────────────────────────────
    cfg = load_config(args.config)

    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    cfg.run_id = args.run_id or _fresh_run_id("rerun_failures")
    logger.info("Run ID: %s", cfg.run_id)

    # ── 3. Determine dataset name and locate the matching DatasetConfig ─────────
    dataset_name = _load_dataset_name(results_dir)
    logger.info("Dataset: %s", dataset_name)

    ds_cfg = next(
        (ds for ds in cfg.datasets if ds.name == dataset_name),
        None,
    )
    if ds_cfg is None:
        logger.error(
            "Dataset '%s' not found in config %s. Available: %s",
            dataset_name,
            args.config,
            [ds.name for ds in cfg.datasets],
        )
        sys.exit(1)

    # ── 4. Build the problem list from the failed rows ─────────────────────────
    # We reconstruct problems directly from the previous results so we don't
    # need to re-load and re-filter the full Arrow dataset.
    problems: list[dict] = [
        {
            "id": r["problem_id"],
            "prompt": r["prompt"],
            "ground_truth": r["ground_truth"],
            "_raw": {},
        }
        for r in failed
    ]
    logger.info("Re-running %d problem(s)", len(problems))
    for p in problems:
        logger.info("  • %s", p["id"])

    # ── 5. Build retriever (if configured) ────────────────────────────────────
    retriever = None
    if cfg.retriever is not None and cfg.retriever.use_retriever:
        logger.info("Building math context retriever …")
        retriever = _build_retriever(cfg.retriever, cfg.datasets)

    # ── 6. Set up output directory ────────────────────────────────────────────
    out_dir = (
        cfg.output_dir
        / cfg.run_id
        / _dataset_slug(ds_cfg.name)
        / _model_slug(cfg.model.model_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Output directory: %s", out_dir)

    # ── 7. Run inference ───────────────────────────────────────────────────────
    system_prompt = _resolve_system_prompt(ds_cfg, cfg)

    backend = build_backend(cfg.model)
    t_start = time.perf_counter()
    try:
        results = run_dataset_benchmark(
            problems=problems,
            ds_cfg=ds_cfg,
            backend=backend,
            model_cfg=cfg.model,
            system_prompt=system_prompt,
            run_cfg=cfg,
            retriever=retriever,
            max_retries=cfg.max_retries,
            out_dir=out_dir,
            run_id=cfg.run_id,
        )
    finally:
        backend.close()

    elapsed = time.perf_counter() - t_start

    # ── 8. Save and report ────────────────────────────────────────────────────
    summary = compute_summary(
        results=results,
        dataset_name=ds_cfg.name,
        model_id=cfg.model.model_id,
        run_id=cfg.run_id,
        elapsed_s=elapsed,
        n_samples=cfg.model.n_samples,
    )
    save_results(results, summary, cfg, ds_cfg.name)

    print(f"\n── Re-run complete  [{cfg.run_id}] ────────────────────────────")
    print(f"  Previously failed : {len(failed)}")
    print(
        f"  Now correct       : {summary['correct']}/{summary['total']}  "
        f"({summary['accuracy']:.1%})"
    )
    print(f"  Errors            : {summary['errors']}")
    print(f"  Wall time         : {elapsed:.1f}s")
    print(f"  Output            : {out_dir}")
    print()

    for r in results:
        status = "CORRECT" if r.get("correct") else "WRONG"
        print(f"  [{status}]  {r['problem_id']}  pred={r.get('model_answer')!r}  gt={r['ground_truth']!r}")
    print()


if __name__ == "__main__":
    main()
