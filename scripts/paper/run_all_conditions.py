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
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path

setup_path()

logger = logging.getLogger(__name__)

HPO_BEST_RUN_ID = "e9c9760f816f46948197519e1c905273"
HPO_S3_PREFIX = (
    "s3://elixirtrials-949678234935-eu-west-2-artifacts"
    f"/mlflow/artifacts/3/{HPO_BEST_RUN_ID}/artifacts"
)


def flush_partial_results(
    all_results: dict[str, dict[str, float | None]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write current results to disk after each condition completes.

    Ensures partial results survive crashes. The file is overwritten
    on each call with the latest cumulative results.
    """
    table = assemble_table2(all_results)
    if metadata:
        table["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table, indent=2))
    logger.info("Flushed %d condition(s) to %s", len(all_results), output_path)


def fetch_best_hpo_adapter(
    dest: Path,
    run_id: str = HPO_BEST_RUN_ID,
    s3_prefix: str = HPO_S3_PREFIX,
    mlflow_tracking_uri: str | None = None,
) -> Path:
    """Fetch the best HPO adapter from S3 or MLflow if not already on disk.

    Tries S3 first, falls back to MLflow CLI. Returns the path to the
    adapter directory containing adapter_config.json.

    Raises:
        FileNotFoundError: If neither S3 nor MLflow download succeeds.
    """
    if (dest / "adapter_config.json").exists():
        logger.info("HPO adapter already present at %s", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("HPO adapter not found at %s — fetching from S3...", dest)

    s3_ok = (
        subprocess.run(
            ["aws", "s3", "cp", f"{s3_prefix}/", f"{dest}/", "--recursive"],
            capture_output=True,
        ).returncode
        == 0
    )

    if not s3_ok:
        logger.info("S3 download failed — trying MLflow CLI...")
        env = None
        if mlflow_tracking_uri:
            import os

            env = {**os.environ, "MLFLOW_TRACKING_URI": mlflow_tracking_uri}
        subprocess.run(
            [
                "uv",
                "run",
                "mlflow",
                "artifacts",
                "download",
                "--run-id",
                run_id,
                "--dst-path",
                str(dest),
            ],
            capture_output=True,
            env=env,
        )

    if not (dest / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Could not fetch HPO adapter. Neither S3 nor MLflow download "
            f"produced adapter_config.json in {dest}"
        )
    logger.info("HPO adapter downloaded to %s", dest)
    return dest


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
    adapter_paths: dict[str, str] | None = None,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Evaluate with a fixed adapter stack (conditions i, iii).

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        adapter_ids: Adapter IDs/paths to stack on top of base model.
            IDs without a corresponding entry in adapter_paths are treated
            as metadata only (not sent to the inference provider).
        provider: InferenceProvider instance.
        adapter_paths: Optional dict mapping adapter_id -> local filesystem
            path. Adapters with paths are loaded into the provider before
            evaluation and unloaded after.
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Directory for per-problem JSONL checkpoints.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    import asyncio

    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack

    paths = adapter_paths or {}

    loop = asyncio.new_event_loop()
    try:
        for aid, path in paths.items():
            loop.run_until_complete(provider.load_adapter(aid, path))

        stack = AdapterStack(
            base_model=model,
            adapter_ids=adapter_ids,
            adapter_paths=paths,
            provider=provider,
        )

        results: dict[str, float | None] = {}
        for bench_id in benchmarks:
            try:
                result = run_benchmark(
                    stack,
                    bench_id,
                    max_samples=max_samples,
                    checkpoint_dir=checkpoint_dir,
                )
                results[bench_id] = result.pass_at_1
            except Exception as exc:
                logger.error("Benchmark %s failed: %s", bench_id, exc, exc_info=True)
                results[bench_id] = None
                continue
    finally:
        for aid in paths:
            try:
                loop.run_until_complete(provider.unload_adapter(aid))
            except Exception:
                logger.warning("Failed to unload adapter %s", aid, exc_info=True)
        loop.close()

    return results


def run_condition_rag(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    corpus_path: Path,
    top_k: int,
    provider: Any,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Evaluate with trajectory-aware RAG prompt augmentation (condition ii).

    Builds a FAISS vector store from the trajectory corpus, then for each
    benchmark problem retrieves top-k trajectory chunks and prepends them
    to the prompt as context.

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        warm_start_adapter: Warm-start LoRA adapter ID.
        corpus_path: Path to JSONL trajectory corpus.
        top_k: Number of chunks to retrieve per query.
        provider: InferenceProvider instance.
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Directory for per-problem JSONL checkpoints.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from model_training.rag_pipeline import (
        RAGConfig,
        _get_encoder,
        build_vector_store,
        query_trajectory_rag,
    )

    config = RAGConfig(top_k=top_k)
    print(f"  Building vector store from {corpus_path}...")
    store = build_vector_store(corpus_path, config)
    encoder = _get_encoder(config.embedding_model)
    print(f"  Built index with {store['n_chunks']} chunks")

    def prompt_augmenter(prompt: str) -> str:
        chunks = query_trajectory_rag(
            query=prompt,
            index=store["index"],
            chunks=store["chunks"],
            encoder=encoder,
            top_k=top_k,
        )
        if not chunks:
            return prompt
        context = "\n---\n".join(chunks)
        return f"# Retrieved trajectory context\n{context}\n\n# Task\n{prompt}"

    stack = AdapterStack(
        base_model=model,
        adapter_ids=[warm_start_adapter],
        adapter_paths={},
        provider=provider,
        prompt_augmenter=prompt_augmenter,
    )

    results: dict[str, float | None] = {}
    for bench_id in benchmarks:
        try:
            result = run_benchmark(
                stack,
                bench_id,
                max_samples=max_samples,
                checkpoint_dir=checkpoint_dir,
            )
            results[bench_id] = result.pass_at_1
        except Exception as exc:
            logger.error("Benchmark %s failed: %s", bench_id, exc, exc_info=True)
            results[bench_id] = None
            continue
    return results


def run_condition_ttt(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    ttt_lr: float,
    ttt_steps: int,
    ttt_mlp_fraction: float,
    provider: Any,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
    """Evaluate with test-time training on MLP layers (condition iv).

    For each benchmark problem, runs inner-loop gradient updates on a
    fraction of MLP layers using the prompt as self-supervised context,
    then generates the completion.

    Args:
        benchmarks: Benchmark IDs to evaluate.
        model: Base model ID.
        warm_start_adapter: Warm-start LoRA adapter ID.
        ttt_lr: Inner-loop learning rate.
        ttt_steps: Number of inner-loop gradient steps.
        ttt_mlp_fraction: Fraction of MLP layers to train.
        provider: InferenceProvider instance.
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Directory for per-problem JSONL checkpoints.

    Returns:
        Dict of {benchmark: pass_at_1}.
    """
    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from model_training.ttt_e2e import TTTConfig, ttt_forward_pass

    ttt_config = TTTConfig(
        mlp_fraction=ttt_mlp_fraction,
        inner_lr=ttt_lr,
        inner_steps=ttt_steps,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model {model} for TTT inner-loop...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    ttt_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    original_sd = {k: v.cpu().clone() for k, v in ttt_model.state_dict().items()}

    def completion_override(prompt: str, max_tokens: int) -> str:
        ttt_model.load_state_dict(original_sd, assign=False)
        result = ttt_forward_pass(
            model=ttt_model,
            tokenizer=tokenizer,
            context=prompt,
            query=prompt,
            config=ttt_config,
        )
        return result["generation"]

    stack = AdapterStack(
        base_model=model,
        adapter_ids=[warm_start_adapter],
        adapter_paths={},
        provider=provider,
        completion_override=completion_override,
    )

    results: dict[str, float | None] = {}
    for bench_id in benchmarks:
        try:
            result = run_benchmark(
                stack,
                bench_id,
                max_samples=max_samples,
                checkpoint_dir=checkpoint_dir,
            )
            results[bench_id] = result.pass_at_1
        except Exception as exc:
            logger.error("Benchmark %s failed: %s", bench_id, exc, exc_info=True)
            results[bench_id] = None
            continue
    return results


def run_condition_rune(
    benchmarks: list[str],
    model: str,
    warm_start_adapter: str,
    hypernet_checkpoint: str,
    device: str,
    provider: Any,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float | None]:
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
        max_samples: Cap on problems per benchmark.
        checkpoint_dir: Directory for per-problem JSONL checkpoints.

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

    results: dict[str, float | None] = {}
    for bench_id in benchmarks:
        try:
            result = run_benchmark(
                stack,
                bench_id,
                max_samples=max_samples,
                checkpoint_dir=checkpoint_dir,
            )
            results[bench_id] = result.pass_at_1
        except Exception as exc:
            logger.error("Benchmark %s failed: %s", bench_id, exc, exc_info=True)
            results[bench_id] = None
            continue
    return results


def assemble_table2(
    all_results: dict[str, dict[str, float | None]],
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
        "--conditions",
        nargs="+",
        default=["i", "ii", "iii", "iv", "v"],
        choices=["i", "ii", "iii", "iv", "v"],
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "livecodebench"],
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter",
        default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument(
        "--adapter-iii",
        type=str,
        default=None,
        help="Path to HPO-tuned QLoRA adapter for Condition (iii)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to JSONL trajectory corpus for Condition (ii) RAG",
    )
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument(
        "--hypernet-checkpoint",
        type=str,
        default=None,
        help="Path to trained hypernetwork checkpoint for Condition (v)",
    )
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--ttt-steps", type=int, default=5)
    parser.add_argument("--ttt-mlp-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap problems per benchmark (useful for quick smoke runs)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/table2.json")
    )
    args = parser.parse_args()

    from inference.factory import get_provider
    from model_training.training_common import (
        mlflow_log_artifact,
        mlflow_log_params,
        setup_mlflow,
    )

    mlflow_ok = setup_mlflow("paper-table2", tracking_uri=None)
    if mlflow_ok:
        import mlflow

        mlflow.start_run(run_name="table2")
        mlflow_log_params(
            {
                "model": args.model,
                "warm_start_adapter": args.warm_start_adapter,
                "benchmarks": ",".join(args.benchmarks),
                "conditions": ",".join(args.conditions),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "rag_top_k": args.rag_top_k,
                "ttt_lr": args.ttt_lr,
                "ttt_steps": args.ttt_steps,
                "ttt_mlp_fraction": args.ttt_mlp_fraction,
            }
        )

    # Auto-fetch HPO adapter for conditions that need it (iii, v)
    hpo_adapter_path: Path | None = None
    needs_hpo = {"iii", "v"} & set(args.conditions)
    if needs_hpo:
        hpo_dest = (
            Path(args.adapter_iii)
            if args.adapter_iii
            else Path("hpo_artifacts/best_diffloss_v1")
        )
        try:
            hpo_adapter_path = fetch_best_hpo_adapter(hpo_dest)
            if not args.adapter_iii:
                args.adapter_iii = str(hpo_adapter_path)
        except FileNotFoundError as exc:
            print(f"  WARNING: Could not fetch HPO adapter: {exc}")
            if "iii" in needs_hpo:
                print("  Condition (iii) will be skipped.")

    provider = get_provider()

    metadata = {
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "benchmarks": args.benchmarks,
        "hypernet_checkpoint": args.hypernet_checkpoint,
    }
    all_results: dict[str, dict[str, float | None]] = {}
    ckpt_base = args.output.parent / "checkpoints"

    for cond in args.conditions:
        print(f"\n{'=' * 60}")
        print(f"Condition ({cond}): {CONDITION_LABELS[cond]}")
        print(f"{'=' * 60}")

        cond_ckpt = ckpt_base / cond
        start = time.time()

        if cond == "i":
            results = run_condition_static(
                args.benchmarks,
                args.model,
                adapter_ids=[args.warm_start_adapter],
                provider=provider,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )

        elif cond == "ii":
            if not args.corpus:
                print("  SKIPPED: --corpus not provided")
                continue
            try:
                results = run_condition_rag(
                    args.benchmarks,
                    args.model,
                    warm_start_adapter=args.warm_start_adapter,
                    corpus_path=args.corpus,
                    top_k=args.rag_top_k,
                    provider=provider,
                    max_samples=args.max_samples,
                    checkpoint_dir=cond_ckpt,
                )
            except ImportError as exc:
                print(f"  SKIPPED: missing dependency for RAG — {exc}")
                continue

        elif cond == "iii":
            if not args.adapter_iii:
                print("  SKIPPED: --adapter-iii not provided and auto-fetch failed")
                continue
            iii_id = "hpo_qlora"
            results = run_condition_static(
                args.benchmarks,
                args.model,
                adapter_ids=[args.warm_start_adapter, iii_id],
                adapter_paths={iii_id: str(args.adapter_iii)},
                provider=provider,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )

        elif cond == "iv":
            results = run_condition_ttt(
                args.benchmarks,
                args.model,
                warm_start_adapter=args.warm_start_adapter,
                ttt_lr=args.ttt_lr,
                ttt_steps=args.ttt_steps,
                ttt_mlp_fraction=args.ttt_mlp_fraction,
                provider=provider,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )

        elif cond == "v":
            if not args.hypernet_checkpoint:
                print("  SKIPPED: --hypernet-checkpoint not provided")
                continue
            results = run_condition_rune(
                args.benchmarks,
                args.model,
                warm_start_adapter=args.warm_start_adapter,
                hypernet_checkpoint=args.hypernet_checkpoint,
                device=args.device,
                provider=provider,
                max_samples=args.max_samples,
                checkpoint_dir=cond_ckpt,
            )

        else:
            continue

        elapsed = time.time() - start
        for bench_id, score in results.items():
            print(f"  {bench_id}: {score:.2%}")
        print(f"  Elapsed: {elapsed:.1f}s")

        all_results[cond] = results
        flush_partial_results(all_results, args.output, metadata)

    print(f"\nTable 2 written to {args.output}")

    if mlflow_ok:
        for cond, scores in all_results.items():
            for bench, score in scores.items():
                mlflow.log_metric(f"{cond}_{bench}_pass_at_1", score)
        mlflow_log_artifact(str(args.output))
        mlflow.end_run()


if __name__ == "__main__":
    main()
