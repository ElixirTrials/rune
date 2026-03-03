"""Evaluation library for adapter benchmarking and fitness scoring.

Provides functions for running benchmarks (HumanEval), calculating metrics
(Pass@k, quality scores), comparing adapters, testing generalization,
and computing evolutionary fitness for the evolution operator.
"""

from evaluation.metrics import (
    calculate_pass_at_k,
    compare_adapters,
    evaluate_fitness,
    run_humaneval_subset,
    score_adapter_quality,
    test_generalization,
)

__all__ = [
    "calculate_pass_at_k",
    "compare_adapters",
    "evaluate_fitness",
    "run_humaneval_subset",
    "score_adapter_quality",
    "test_generalization",
]
