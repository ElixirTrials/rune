"""Benchmark: 3 challenging tasks through the full Rune pipeline.

Runs event-sourced ledger, regex engine, and document store tasks through
run_phased_pipeline() and evaluates decomposition quality, code output,
and test pass rates.

Usage:
    uv run python scripts/benchmark_challenging.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Configure inference to use transformers with gemma (matches Sakana checkpoint)
MODEL_NAME = "google/gemma-2-2b-it"
os.environ.setdefault("INFERENCE_PROVIDER", "transformers")
os.environ.setdefault("TRANSFORMERS_MODEL_NAME", MODEL_NAME)
os.environ.setdefault("RUNE_MODEL", MODEL_NAME)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import setup_path

setup_path()

from shared.hardware import get_best_device
from shared.sandbox import count_test_results, get_sandbox_backend, has_unittest_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_challenging")

DEVICE = get_best_device()

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: list[dict[str, str]] = [
    {
        "name": "Event-Sourced Bank Ledger",
        "description": (
            "Build an event-sourced bank ledger in Python. "
            "LedgerEvent dataclass with fields: event_id (uuid), account_id, "
            "event_type (credit/debit/transfer), amount (Decimal), timestamp, metadata dict. "
            "EventStore class backed by sqlite3 with append-only writes, "
            "replay_events(account_id) returning ordered list. "
            "Ledger class with create_account, credit, debit, transfer (atomic — "
            "debit source + credit dest in one transaction, raise on insufficient funds), "
            "get_balance (replay events to compute), get_balance_at (point-in-time replay). "
            "All monetary amounts must use decimal.Decimal. "
            "Include 15+ unittest tests covering: basic credit/debit, transfer success "
            "and insufficient funds, balance computation, point-in-time balance, "
            "concurrent account isolation, empty account balance, event ordering."
        ),
    },
    {
        "name": "Regex Engine with Backtracking",
        "description": (
            "Build a regular expression engine in Python from scratch (no re module). "
            "Lexer that tokenizes regex strings into token types: LITERAL, DOT, STAR, "
            "PLUS, QUESTION, LPAREN, RPAREN, PIPE, LBRACKET, RBRACKET, CARET, BACKSLASH. "
            "Parser that builds an AST with nodes: Literal, Dot, Concat, Alternation, "
            "Quantifier (greedy star/plus/question), CharClass (e.g. [a-z], [^0-9]). "
            "NFA construction via Thompson's algorithm with State and Fragment classes. "
            "RegexEngine class with compile(pattern), match(text) -> bool (anchored), "
            "search(text) -> Match|None (first occurrence with start/end), "
            "findall(text) -> list[str]. "
            "Support: literal chars, dot, *, +, ?, alternation |, character classes [abc], "
            "ranges [a-z], negated classes [^abc], parenthesized groups. "
            "Include 15+ unittest tests covering: literal match, dot, quantifiers, "
            "alternation, character classes, negated classes, groups, findall, "
            "no-match cases, empty pattern, complex nested patterns."
        ),
    },
    {
        "name": "In-Memory Document Store",
        "description": (
            "Build an in-memory document store with query language in Python. "
            "Document class wrapping a dict with auto-generated _id (uuid). "
            "Collection class with insert_one, insert_many, find_one, find (with query), "
            "update_one, delete_one, count methods. "
            "QueryEngine supporting operators: $eq, $ne, $gt, $gte, $lt, $lte, "
            "$in, $nin, $exists, $regex (using re module), "
            "logical: $and, $or, $not, $nor. Nested field access via dot notation. "
            "Secondary indexes: create_index(field) builds a dict-based index, "
            "find() automatically uses index when querying indexed fields. "
            "Aggregation pipeline with stages: $match, $group (with $sum/$avg/$min/$max), "
            "$sort, $limit, $project. "
            "Include 15+ unittest tests covering: CRUD operations, each query operator, "
            "logical operators, nested field queries, index creation and usage, "
            "aggregation pipeline stages, edge cases (empty collection, no match)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def evaluate_code(code: str, timeout: int = 60) -> dict[str, Any]:
    """Run generated code in sandbox and collect test results."""
    if not code or not code.strip():
        return {
            "lines": 0,
            "has_tests": False,
            "tests_ran": False,
            "passed": 0,
            "total": 0,
            "exit_code": 1,
            "error": "empty code",
        }

    script = code
    if has_unittest_classes(script) and "unittest.main" not in script:
        script += (
            "\n\nimport unittest\nif __name__ == '__main__':\n    unittest.main()\n"
        )

    backend = get_sandbox_backend()
    result = backend.run(script, timeout=timeout)

    passed, total = count_test_results(result.stdout, result.stderr)

    return {
        "lines": len(code.splitlines()),
        "has_tests": has_unittest_classes(code),
        "tests_ran": total > 0,
        "passed": passed,
        "total": total,
        "exit_code": result.exit_code,
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    from model_training.sakana_d2l import download_checkpoint
    from rune_runner import run_phased_pipeline

    # Download checkpoint once
    checkpoint_path = str(download_checkpoint(variant="gemma_demo"))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {DEVICE}")
    print(f"Tasks: {len(TASKS)}")
    print()

    results: list[dict[str, Any]] = []

    for ti, task in enumerate(TASKS):
        print(f"\n{'=' * 70}")
        print(f"  Task {ti + 1}/{len(TASKS)}: {task['name']}")
        print(f"{'=' * 70}\n")

        t0 = time.time()

        pipeline_result = await run_phased_pipeline(
            project_prompt=task["description"],
            checkpoint_path=checkpoint_path,
            base_model_id=MODEL_NAME,
            device=DEVICE,
            max_phase_iterations=3,
        )

        elapsed = time.time() - t0

        # Extract results
        final_code = pipeline_result.get("accumulated_code", "")
        subtasks = pipeline_result.get("subtasks", [])
        phases = pipeline_result.get("phases", {})
        decompose_info = phases.get("decompose", {})
        code_info = phases.get("code", {})
        integrate_info = phases.get("integrate", {})

        # Independent code evaluation
        eval_result = evaluate_code(final_code)

        task_result = {
            "task": task["name"],
            "elapsed_s": round(elapsed, 1),
            "decompose": {
                "subtask_count": len(subtasks),
                "subtask_names": subtasks,
                "score": decompose_info.get("best_score", 0),
                "iterations": decompose_info.get("iterations", 0),
            },
            "code": {
                "final_lines": eval_result["lines"],
                "has_tests": eval_result["has_tests"],
                "tests_ran": eval_result["tests_ran"],
                "tests_passed": eval_result["passed"],
                "tests_total": eval_result["total"],
                "exit_code": eval_result["exit_code"],
                "code_phase_passed": code_info.get("passed", 0),
                "code_phase_total": code_info.get("total", 0),
            },
            "integrate": {
                "tests_passed": integrate_info.get("tests_passed", False),
                "score": integrate_info.get("best_score", 0),
                "iterations": integrate_info.get("iterations", 0),
            },
            "pipeline": {
                "total_iterations": pipeline_result.get("total_iterations", 0),
                "final_tests_passed": pipeline_result.get("final_tests_passed", False),
            },
        }

        results.append(task_result)

        # Print per-task summary
        print(f"\n  --- {task['name']} Results ---")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Subtasks: {len(subtasks)} — {subtasks}")
        print(f"  Decompose score: {decompose_info.get('best_score', 0):.2f}")
        print(
            f"  Code: {eval_result['lines']} lines, "
            f"{eval_result['passed']}/{eval_result['total']} tests passed"
        )
        print(f"  Integration score: {integrate_info.get('best_score', 0):.2f}")
        print(f"  Pipeline passed: {pipeline_result.get('final_tests_passed', False)}")

    # -------------------------------------------------------------------
    # Aggregate summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  AGGREGATE SUMMARY")
    print(f"{'=' * 70}\n")

    header = (
        f"  {'Task':<30} {'Subtasks':>8} {'Lines':>6} "
        f"{'Tests':>12} {'IntScore':>9} {'Pass':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_passed = 0
    total_tests = 0
    total_lines = 0

    for r in results:
        c = r["code"]
        i = r["integrate"]
        p = r["pipeline"]
        tp = c["tests_passed"]
        tt = c["tests_total"]
        total_passed += tp
        total_tests += tt
        total_lines += c["final_lines"]
        pass_str = "YES" if p["final_tests_passed"] else "no"
        print(
            f"  {r['task']:<30} {r['decompose']['subtask_count']:>8} "
            f"{c['final_lines']:>6} {tp:>5}/{tt:<5} "
            f"{i['score']:>8.2f} {pass_str:>5}"
        )

    print()
    print(f"  Total lines generated: {total_lines}")
    print(f"  Total tests: {total_passed}/{total_tests} passed")
    print(
        f"  Tasks with pipeline pass: "
        f"{sum(1 for r in results if r['pipeline']['final_tests_passed'])}/{len(results)}"
    )

    # Assessment
    print("\n  Assessment:")
    for r in results:
        c = r["code"]
        status = "PASS" if r["pipeline"]["final_tests_passed"] else "FAIL"
        quality = "good" if c["final_lines"] > 50 else "thin"
        if c["tests_total"] == 0:
            test_quality = "no tests ran"
        elif c["tests_passed"] == c["tests_total"]:
            test_quality = "all tests passed"
        elif c["tests_passed"] > 0:
            test_quality = f"{c['tests_passed']}/{c['tests_total']} tests passed"
        else:
            test_quality = "no tests passed"
        print(
            f"    [{status}] {r['task']}: {quality} code ({c['final_lines']}L), {test_quality}"
        )

    # Save results
    out_path = Path(__file__).parent / "benchmark_challenging_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
