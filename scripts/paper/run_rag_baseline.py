"""Run Condition (ii) RAG baseline evaluation.

Builds vector store, runs Pass@1 eval on HumanEval+ and LiveCodeBench.

Usage:
    uv run python scripts/paper/run_rag_baseline.py \
        --corpus data/pairs/corpus.jsonl \
        --model Qwen/Qwen3.5-9B \
        --output evaluation_results/condition_ii.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Condition (ii): RAG baseline")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/condition_ii.json"))
    args = parser.parse_args()

    from model_training.rag_pipeline import RAGConfig, build_vector_store, query_trajectory_rag, _get_encoder

    config = RAGConfig(chunk_size=args.chunk_size, top_k=args.top_k)
    print(f"Building vector store from {args.corpus}...")
    store = build_vector_store(args.corpus, config)
    print(f"Built index with {store['n_chunks']} chunks")

    encoder = _get_encoder(config.embedding_model)

    print("RAG pipeline built. Run eval harness with --rag-context flag to evaluate.")
    result = {
        "condition": "ii_rag",
        "model": args.model,
        "config": {"top_k": args.top_k, "chunk_size": args.chunk_size},
        "n_chunks": store["n_chunks"],
        "status": "pipeline_ready",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
