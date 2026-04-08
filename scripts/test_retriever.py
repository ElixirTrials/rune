"""Smoke-test the MathContextRetriever against 5 problems from the dataset.

Loads retriever config from the benchmark YAML, builds/loads the index,
queries with 5 problems drawn from olym_math_easy, and writes a structured
JSON report to retriever_test_results.json.

Usage::

    cd /workspace/rune
    uv run python scripts/test_retriever.py
    uv run python scripts/test_retriever.py --config path/to/other.yaml --n 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── make sure libs are importable ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend(
    [
        str(_REPO_ROOT / "libs" / "evaluation" / "src"),
        str(_REPO_ROOT / "libs" / "shared" / "src"),
    ]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_retriever")


def _count_tokens(text: str) -> int:
    """Rough token count: one token ≈ 4 characters (standard GPT heuristic)."""
    return max(1, len(text) // 4)


def main(config_path: str, n_problems: int, out_path: str) -> None:
    from evaluation.config import load_config
    from evaluation.benchmark_runner import _build_retriever
    from evaluation.utils import load_problems

    logger.info("Loading config from %s", config_path)
    cfg = load_config(config_path)

    if cfg.retriever is None or not cfg.retriever.use_retriever:
        logger.error("No retriever configured in %s — aborting", config_path)
        sys.exit(1)

    # ── Build / load the retriever index ──────────────────────────────────────
    logger.info("Initialising retriever …")
    t0 = time.perf_counter()
    retriever = _build_retriever(cfg.retriever, cfg.datasets)
    index_elapsed = time.perf_counter() - t0
    logger.info("Retriever ready in %.1fs", index_elapsed)

    # ── Load dataset problems ─────────────────────────────────────────────────
    ds_cfg = cfg.datasets[0]
    logger.info("Loading problems from dataset '%s'", ds_cfg.name)
    all_problems = load_problems(ds_cfg)
    problems = all_problems[:n_problems]
    logger.info("Using %d / %d problems", len(problems), len(all_problems))

    # ── Query retriever for each problem ──────────────────────────────────────
    results = []
    for i, prob in enumerate(problems, 1):
        prob_id = prob["id"]
        query_text = prob["prompt"]

        logger.info("[%d/%d] Querying retriever for problem %s …", i, len(problems), prob_id)
        t_q = time.perf_counter()
        examples = retriever.query(query_text)
        query_elapsed = time.perf_counter() - t_q

        formatted_context = retriever.format_context(examples)

        retrieved = []
        for rank, ex in enumerate(examples, 1):
            problem_tokens = _count_tokens(ex.problem)
            solution_tokens = _count_tokens(ex.solution)
            retrieved.append(
                {
                    "rank": rank,
                    "source": ex.source,
                    "similarity": round(ex.similarity, 6),
                    "metadata": ex.metadata,
                    "problem": ex.problem,
                    "solution": ex.solution,
                    "problem_tokens": problem_tokens,
                    "solution_tokens": solution_tokens,
                    "total_tokens": problem_tokens + solution_tokens,
                }
            )

        total_retrieved_tokens = sum(r["total_tokens"] for r in retrieved)
        context_tokens = _count_tokens(formatted_context)

        results.append(
            {
                "problem_id": prob_id,
                "query_text": query_text,
                "query_tokens": _count_tokens(query_text),
                "ground_truth": prob["ground_truth"],
                "query_elapsed_s": round(query_elapsed, 4),
                "n_retrieved": len(retrieved),
                "total_retrieved_tokens": total_retrieved_tokens,
                "formatted_context_tokens": context_tokens,
                "retrieved": retrieved,
            }
        )

        logger.info(
            "  → %d examples  total_tokens=%d  similarity range [%.3f–%.3f]  %.3fs",
            len(retrieved),
            total_retrieved_tokens,
            min(r["similarity"] for r in retrieved) if retrieved else 0.0,
            max(r["similarity"] for r in retrieved) if retrieved else 0.0,
            query_elapsed,
        )

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "config_path": str(Path(config_path).resolve()),
        "dataset": ds_cfg.name,
        "embedding_model": cfg.retriever.embedding_model,
        "top_k": cfg.retriever.top_k,
        "tir_top_k": cfg.retriever.tir_top_k,
        "similarity_threshold": cfg.retriever.similarity_threshold,
        "index_load_elapsed_s": round(index_elapsed, 2),
        "n_problems": len(results),
        "results": results,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved → %s", out_file.resolve())

    # ── Quick summary table ───────────────────────────────────────────────────
    print("\n── Retriever test summary ──────────────────────────────────")
    print(f"{'Problem ID':<30}  {'Retrieved':>9}  {'Tokens':>8}  {'Top sim':>8}  {'Time':>7}")
    print("─" * 70)
    for r in results:
        top_sim = max((x["similarity"] for x in r["retrieved"]), default=0.0)
        print(
            f"{r['problem_id']:<30}  {r['n_retrieved']:>9}  "
            f"{r['total_retrieved_tokens']:>8}  {top_sim:>8.3f}  {r['query_elapsed_s']:>6.3f}s"
        )
    print(f"\nOutput: {out_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MathContextRetriever with dataset prompts")
    parser.add_argument(
        "--config",
        default="libs/evaluation/src/evaluation/configs/qwen3_5_olym_easy.yaml",
        help="Path to benchmark YAML config (default: qwen3_5_olym_easy.yaml)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of problems to query (default: 5)",
    )
    parser.add_argument(
        "--out",
        default="retriever_test_results.json",
        help="Output JSON file path (default: retriever_test_results.json)",
    )
    args = parser.parse_args()
    main(args.config, args.n, args.out)
