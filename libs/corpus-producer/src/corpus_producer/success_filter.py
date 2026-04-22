"""Pass@1 success filter for phase corpus artifacts.

Calls the benchmark harness to evaluate the integrated final code and
marks every artifact in the run with the result. Keeps artifacts according
to the filtering policy:

  Pass@1 == 1.0  ->  keep ALL phase artifacts (they all contributed to success)
  Pass@1 == 0.0  ->  keep ONLY diagnose artifacts where diagnose->repair SUCCEEDED
  Any other rate ->  discard (partial pass means the code is not fully correct)

This implements the STaR-style self-distillation principle: only traces that
produced correct outputs are used as training data.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from corpus_producer.models import PhaseArtifact

logger = logging.getLogger(__name__)

# Module-level sentinel so unittest.mock.patch can find the name.
# Replaced at import time if evaluation.benchmarks is available (GPU env);
# patched in tests via patch("corpus_producer.success_filter.run_benchmark").
run_benchmark: Optional[Callable[..., Any]]
try:
    from evaluation.benchmarks import (  # type: ignore[assignment,no-redef,import-not-found]
        run_benchmark,
    )
except ImportError:  # CPU-only CI — will be patched in tests
    run_benchmark = None


def filter_artifacts(
    artifacts: list[PhaseArtifact],
    final_code: str,
    benchmark: str,
    problem_id: str,
) -> list[PhaseArtifact]:
    """Evaluate Pass@1 and return only training-worthy artifacts.

    Uses the module-level ``run_benchmark`` reference which is patchable via
    ``unittest.mock.patch("corpus_producer.success_filter.run_benchmark")``.

    Args:
        artifacts: All per-phase artifacts from a single pipeline run.
        final_code: Integrated output (Phase 4) to evaluate.
        benchmark: Benchmark identifier.
        problem_id: Problem identifier used to scope the benchmark run.

    Returns:
        Filtered list of PhaseArtifact with ``pass_at_1`` field set.
    """
    if run_benchmark is None:
        raise RuntimeError(
            "evaluation.benchmarks.run_benchmark is unavailable; patch "
            "corpus_producer.success_filter.run_benchmark in tests."
        )
    result = run_benchmark(
        model_adapter_stack=("__direct_eval__", []),
        benchmark_id=benchmark,
        problem_ids=[problem_id],
        max_samples=1,
    )

    verdict = result.per_problem.get(problem_id)
    passed = verdict.passed if verdict is not None else False

    logger.info(
        "Pass@1 for %s/%s: %s",
        benchmark,
        problem_id,
        "PASS" if passed else "FAIL",
    )

    kept: list[PhaseArtifact] = []

    if passed:
        # All phases contributed to a successful outcome.
        for art in artifacts:
            art.pass_at_1 = True
            kept.append(art)
    else:
        # Only keep diagnose artifacts where the repair succeeded.
        # The repair success flag is encoded in the artifact metadata
        # by the pipeline runner when the diagnose->repair loop recovers.
        for art in artifacts:
            art.pass_at_1 = False
            repair_ok = art.metadata.get("repair_succeeded") == "true"
            if art.phase == "diagnose" and repair_ok:
                kept.append(art)

    return kept
