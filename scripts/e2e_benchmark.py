"""E2E Benchmark: Run Rune pipeline on multiple coding tasks and report performance.

Tasks inspired by popular coding benchmarks (HumanEval, MBPP, SWE-bench):
  1. LRU Cache — classic data structures (HumanEval-style)
  2. Matrix Operations Library — numerical computing (MBPP-style)
  3. URL Shortener — web/systems design (SWE-bench-style)

Usage:
    uv run scripts/e2e_benchmark.py
    uv run scripts/e2e_benchmark.py --tasks 1,2     # run specific tasks
    uv run scripts/e2e_benchmark.py --max-phase-iterations 3
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e_benchmark")

MODEL_NAME = "google/gemma-2-2b-it"

from shared.hardware import get_best_device

DEVICE = get_best_device()

# ---------------------------------------------------------------------------
# Benchmark tasks
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "lru_cache",
        "name": "LRU Cache with TTL",
        "category": "data-structures",
        "prompt": (
            "Build a Python LRU (Least Recently Used) cache library. Requirements:\n\n"
            "1. LRUCache class with configurable max_size (default 128). Uses a "
            "doubly-linked list + hash map for O(1) get/put. Supports get(key), "
            "put(key, value), delete(key), clear(), and __len__.\n\n"
            "2. TTL support: optional ttl_seconds parameter on put(). Expired entries "
            "are lazily evicted on access and eagerly cleaned by a purge() method.\n\n"
            "3. Statistics: hits, misses, evictions counters accessible via stats() "
            "returning a dict.\n\n"
            "4. Decorator: @lru_cached(max_size, ttl) that wraps any function with "
            "the cache. Must handle unhashable arguments gracefully (skip caching).\n\n"
            "5. Thread safety: all public methods protected by a threading.Lock.\n\n"
            "6. Include a unittest test suite with at least 12 test methods covering: "
            "basic get/put, eviction at capacity, TTL expiry, stats tracking, "
            "decorator usage, thread safety with concurrent access, edge cases "
            "(empty cache, duplicate keys, zero capacity)."
        ),
    },
    {
        "id": "matrix_ops",
        "name": "Matrix Operations Library",
        "category": "numerical-computing",
        "prompt": (
            "Build a pure-Python matrix operations library (no numpy). Requirements:\n\n"
            "1. Matrix class storing data as list[list[float]]. Constructor validates "
            "rectangular shape. Properties: rows, cols, shape. Supports __repr__, "
            "__eq__, __getitem__(row, col), and __setitem__.\n\n"
            "2. Arithmetic: __add__, __sub__, __mul__ (scalar and matrix multiply), "
            "__neg__, __truediv__ (scalar). All validate dimensions.\n\n"
            "3. Class methods: Matrix.zeros(r, c), Matrix.identity(n), "
            "Matrix.from_list(nested_list).\n\n"
            "4. Operations: transpose(), determinant() (recursive cofactor expansion "
            "for any NxN), inverse() (via adjugate/det, raises ValueError if singular), "
            "trace().\n\n"
            "5. Row operations: row_echelon_form() returning upper triangular form, "
            "rank() computed from REF.\n\n"
            "6. Include a unittest test suite with at least 12 test methods covering: "
            "construction and validation, arithmetic operations, identity/zeros, "
            "determinant (2x2, 3x3, singular), inverse and inverse-of-inverse, "
            "transpose, rank, edge cases (1x1, non-square, dimension mismatch errors)."
        ),
    },
    {
        "id": "url_shortener",
        "name": "URL Shortener Service",
        "category": "web-systems",
        "prompt": (
            "Build a Python URL shortener library (no web framework needed). Requirements:\n\n"
            "1. URLStore class backed by an in-memory dict with optional SQLite "
            "persistence via sqlite3. Schema: urls(short_code TEXT PRIMARY KEY, "
            "original_url TEXT NOT NULL, created_at TEXT, click_count INTEGER DEFAULT 0, "
            "expires_at TEXT). Supports CRUD.\n\n"
            "2. ShortenerService class: shorten(url, custom_code=None, ttl_hours=None) "
            "generates a unique 6-char alphanumeric code (base62), resolve(code) returns "
            "original URL and increments click_count, raises KeyError if expired/missing. "
            "stats(code) returns click_count and created_at.\n\n"
            "3. Validation: reject invalid URLs (must start with http:// or https://), "
            "reject custom codes that collide with existing entries, auto-retry on "
            "random code collision (up to 5 attempts).\n\n"
            "4. Bulk operations: shorten_batch(urls) returns list of codes, "
            "cleanup_expired() removes all expired entries and returns count removed.\n\n"
            "5. Include a unittest test suite with at least 12 test methods covering: "
            "shorten and resolve round-trip, custom codes, URL validation, expiry, "
            "click counting, bulk operations, collision handling, SQLite persistence "
            "round-trip, edge cases (empty URL, very long URL, special characters)."
        ),
    },
]


def run_single_task(
    task: dict,
    checkpoint_path: str,
    max_phase_iterations: int,
    tmpdir: str,
) -> dict:
    """Run the pipeline on a single task and collect results."""
    from inference.factory import _clear_cache
    from rune_runner import run_phased_pipeline

    _clear_cache()

    os.environ["INFERENCE_PROVIDER"] = "transformers"
    os.environ["TRANSFORMERS_MODEL_NAME"] = MODEL_NAME
    os.environ["TRANSFORMERS_DEVICE"] = DEVICE

    print(f"\n{'#' * 70}")
    print(f"  TASK: {task['name']} ({task['category']})")
    print(f"  Prompt: {len(task['prompt'])} chars")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    result = asyncio.run(
        run_phased_pipeline(
            project_prompt=task["prompt"],
            max_iterations=10,
            checkpoint_path=checkpoint_path,
            base_model_id=MODEL_NAME,
            device=DEVICE,
            population_size=2,
            max_phase_iterations=max_phase_iterations,
        )
    )

    elapsed = time.time() - t0

    # Extract metrics
    phases = result.get("phases", {})
    decompose = phases.get("decompose", {})
    plan = phases.get("plan", {})
    code = phases.get("code", {})
    integrate = phases.get("integrate", {})

    subtasks = decompose.get("subtasks", [])
    code_passed = code.get("passed", 0)
    code_total = code.get("total", 0)

    # Count total adapters generated
    n_adapters = len(result.get("adapters", []))

    # Evolution stats
    evolution = result.get("evolution", {})
    phase_iters = evolution.get("phase_iterations", {})

    summary = {
        "task_id": task["id"],
        "task_name": task["name"],
        "category": task["category"],
        "elapsed_seconds": round(elapsed, 1),
        "session_id": result["session_id"],
        "total_iterations": result["total_iterations"],
        "n_subtasks": len(subtasks),
        "subtask_names": [s["name"] for s in subtasks],
        "n_plans": len(plan.get("plans", {})),
        "code_passed": code_passed,
        "code_total": code_total,
        "code_pass_rate": round(code_passed / max(code_total, 1), 2),
        "integration_passed": integrate.get("tests_passed", False),
        "n_adapters": n_adapters,
        "phase_iterations": phase_iters,
        "decompose_score": decompose.get("best_score", 0),
        "plan_score": plan.get("best_score", 0),
        "integrate_score": integrate.get("best_score", 0),
        "final_code_length": len(result.get("accumulated_code", "")),
    }

    return summary


def print_report(results: list[dict]) -> None:
    """Print a comprehensive performance report."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n  --- {r['task_name']} ({r['category']}) ---")
        print(f"    Session:          {r['session_id']}")
        print(f"    Time:             {r['elapsed_seconds']}s")
        print(f"    Subtasks:         {r['n_subtasks']} ({', '.join(r['subtask_names'][:5])})")
        print(f"    Plans generated:  {r['n_plans']}")
        print(f"    Code pass rate:   {r['code_passed']}/{r['code_total']} ({r['code_pass_rate']:.0%})")
        print(f"    Integration:      {'PASS' if r['integration_passed'] else 'FAIL'}")
        print(f"    Adapters:         {r['n_adapters']}")
        print(f"    Phase iterations: {r['phase_iterations']}")
        print(f"    Final code:       {r['final_code_length']} chars")
        print(f"    Scores:           decompose={r['decompose_score']:.2f} plan={r['plan_score']:.2f} integrate={r['integrate_score']:.2f}")

    # Aggregate stats
    print("\n  --- AGGREGATE ---")
    total_time = sum(r["elapsed_seconds"] for r in results)
    avg_time = total_time / len(results)
    total_code_pass = sum(r["code_passed"] for r in results)
    total_code_total = sum(r["code_total"] for r in results)
    integration_pass = sum(1 for r in results if r["integration_passed"])
    total_adapters = sum(r["n_adapters"] for r in results)
    avg_subtasks = sum(r["n_subtasks"] for r in results) / len(results)

    print(f"    Tasks run:           {len(results)}")
    print(f"    Total time:          {total_time:.0f}s ({avg_time:.0f}s avg per task)")
    print(f"    Avg subtasks:        {avg_subtasks:.1f}")
    print(f"    Code subtask pass:   {total_code_pass}/{total_code_total} ({total_code_pass/max(total_code_total,1):.0%})")
    print(f"    Integration pass:    {integration_pass}/{len(results)} ({integration_pass/len(results):.0%})")
    print(f"    Total adapters:      {total_adapters}")

    # Performance assessment
    print("\n  --- ASSESSMENT ---")
    if total_code_total > 0:
        code_rate = total_code_pass / total_code_total
        if code_rate >= 0.7:
            print(f"    Code generation:     STRONG ({code_rate:.0%} subtasks passing)")
        elif code_rate >= 0.4:
            print(f"    Code generation:     MODERATE ({code_rate:.0%} subtasks passing)")
        else:
            print(f"    Code generation:     WEAK ({code_rate:.0%} subtasks passing)")

    int_rate = integration_pass / len(results)
    if int_rate >= 0.7:
        print(f"    Integration:         STRONG ({int_rate:.0%} tasks integrated)")
    elif int_rate >= 0.3:
        print(f"    Integration:         MODERATE ({int_rate:.0%} tasks integrated)")
    else:
        print(f"    Integration:         WEAK ({int_rate:.0%} tasks integrated)")

    avg_decompose = sum(r["decompose_score"] for r in results) / len(results)
    if avg_decompose >= 0.6:
        print(f"    Decomposition:       STRONG (avg score {avg_decompose:.2f})")
    elif avg_decompose >= 0.3:
        print(f"    Decomposition:       MODERATE (avg score {avg_decompose:.2f})")
    else:
        print(f"    Decomposition:       WEAK (avg score {avg_decompose:.2f})")

    print(f"\n{'=' * 70}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rune E2E Benchmark")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task numbers to run (1-indexed, e.g. 1,2)",
    )
    parser.add_argument(
        "--max-phase-iterations",
        type=int,
        default=3,
        help="Max evolutionary iterations per phase (default: 3)",
    )
    args = parser.parse_args()

    task_indices = list(range(len(TASKS)))
    if args.tasks:
        task_indices = [int(t) - 1 for t in args.tasks.split(",")]

    selected_tasks = [TASKS[i] for i in task_indices]

    print("=" * 70)
    print("  RUNE E2E BENCHMARK")
    print("=" * 70)
    print(f"  Base model:           {MODEL_NAME}")
    print(f"  Device:               {DEVICE}")
    print(f"  Tasks:                {len(selected_tasks)}")
    print(f"  Max phase iterations: {args.max_phase_iterations}")
    print(f"  HF token:             {'set' if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') else 'NOT SET'}")

    # Download checkpoint once
    from model_training.sakana_d2l import download_checkpoint

    print("\n  Downloading Sakana checkpoint...")
    checkpoint_path = str(download_checkpoint(variant="gemma_demo"))
    print(f"  Checkpoint: {checkpoint_path}")

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for task in selected_tasks:
            try:
                summary = run_single_task(
                    task=task,
                    checkpoint_path=checkpoint_path,
                    max_phase_iterations=args.max_phase_iterations,
                    tmpdir=tmpdir,
                )
                results.append(summary)
            except Exception as e:
                logger.exception("Task %s failed", task["id"])
                results.append({
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "category": task["category"],
                    "elapsed_seconds": 0,
                    "session_id": "FAILED",
                    "total_iterations": 0,
                    "n_subtasks": 0,
                    "subtask_names": [],
                    "n_plans": 0,
                    "code_passed": 0,
                    "code_total": 0,
                    "code_pass_rate": 0,
                    "integration_passed": False,
                    "n_adapters": 0,
                    "phase_iterations": {},
                    "decompose_score": 0,
                    "plan_score": 0,
                    "integrate_score": 0,
                    "final_code_length": 0,
                    "error": str(e),
                })

    print_report(results)

    # Save JSON results
    results_path = Path(__file__).parent / "benchmark_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
