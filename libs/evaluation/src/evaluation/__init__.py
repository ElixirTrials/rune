"""Evaluation library for adapter benchmarking and fitness scoring.

Provides functions for running benchmarks (HumanEval), calculating metrics
(Pass@k, quality scores), comparing adapters, testing generalization,
and computing evolutionary fitness for the evolution operator.

Also exposes the config-driven :class:`BenchmarkRunner` for large-scale
dataset evaluation across any ``InferenceProvider`` backend.
"""

from evaluation.benchmark_runner import (
    Backend,
    BenchmarkRunner,
    GenerationOutput,
    InferenceProviderBackend,
    VLLMBackend,
    build_backend,
    run_dataset_benchmark,
)
from evaluation.config import (
    BenchmarkRunConfig,
    DatasetConfig,
    DATASET_REGISTRY,
    ModelConfig,
    load_config,
)
from evaluation.metrics import (
    calculate_pass_at_k,
    compare_adapters,
    evaluate_fitness,
    run_humaneval_subset,
    run_kill_switch_gate,
    score_adapter_quality,
    test_generalization,
)
from evaluation.ood_benchmark import compute_generalization_delta, run_ood_benchmark
from evaluation.utils import (
    SCORERS,
    compute_summary,
    load_problems,
    render_template,
    save_results,
    _integer_in_range as integer_in_range,
)

__all__ = [
    # benchmark_runner
    "Backend",
    "BenchmarkRunner",
    "GenerationOutput",
    "InferenceProviderBackend",
    "VLLMBackend",
    "build_backend",
    "run_dataset_benchmark",
    # config
    "BenchmarkRunConfig",
    "DatasetConfig",
    "DATASET_REGISTRY",
    "ModelConfig",
    "load_config",
    # utils
    "SCORERS",
    "compute_summary",
    "integer_in_range",
    "load_problems",
    "render_template",
    "save_results",
    # metrics
    "calculate_pass_at_k",
    "compare_adapters",
    "evaluate_fitness",
    "run_humaneval_subset",
    "run_kill_switch_gate",
    "score_adapter_quality",
    "test_generalization",
    # ood_benchmark
    "compute_generalization_delta",
    "run_ood_benchmark",
]
