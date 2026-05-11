"""Run Condition (ii) RAG baseline evaluation.

Builds vector store from trajectory corpus, retrieves relevant chunks per
benchmark problem, prepends them to prompts, and runs Pass@1 evaluation.

Usage:
    uv run python scripts/paper/run_rag_baseline.py \
        --corpus data/pairs/corpus.jsonl \
        --model Qwen/Qwen3.5-9B \
        --benchmarks humaneval livecodebench \
        --output evaluation_results/condition_ii.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bootstrap import setup_path

setup_path()

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_WARM_START = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Condition (ii): RAG baseline")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--warm-start-adapter",
        default=DEFAULT_WARM_START,
        help="Warm-start LoRA for substrate (DeltaCoder)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "livecodebench"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_results/condition_ii.json")
    )
    args = parser.parse_args()

    from evaluation.benchmarks import run_benchmark
    from evaluation.benchmarks.adapter_stack import AdapterStack
    from inference.factory import get_provider
    from model_training.rag_pipeline import (
        RAGConfig,
        _get_encoder,
        build_vector_store,
        query_trajectory_rag,
    )

    config = RAGConfig(chunk_size=args.chunk_size, top_k=args.top_k)
    print(f"Building vector store from {args.corpus}...")
    store = build_vector_store(args.corpus, config)
    encoder = _get_encoder(config.embedding_model)
    print(f"Built index with {store['n_chunks']} chunks")

    def prompt_augmenter(prompt: str) -> str:
        chunks = query_trajectory_rag(
            query=prompt,
            index=store["index"],
            chunks=store["chunks"],
            encoder=encoder,
            top_k=args.top_k,
        )
        if not chunks:
            return prompt
        context = "\n---\n".join(chunks)
        return f"# Retrieved trajectory context\n{context}\n\n# Task\n{prompt}"

    provider = get_provider()
    stack = AdapterStack(
        base_model=args.model,
        adapter_ids=[args.warm_start_adapter],
        adapter_paths={},
        provider=provider,
        prompt_augmenter=prompt_augmenter,
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
        "condition": "ii_rag",
        "model": args.model,
        "warm_start_adapter": args.warm_start_adapter,
        "config": {
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "embedding_model": config.embedding_model,
        },
        "n_chunks": store["n_chunks"],
        "benchmarks": all_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
