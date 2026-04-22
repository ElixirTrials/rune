"""STaR-style rationalization fallback for thin oracle bins.

When a bin has fewer than MIN_EXAMPLES_PER_BIN examples after the primary
self-distillation pass, this module re-runs the pipeline for the *failing*
problems, augmenting the prompt with ground-truth test outcome hints. This
gives the model extra signal to produce a passing trace, which is then
filtered, binned, and added to the manifest — tagged with rationalized=True.

Reference: Zelikman et al. 2022 (STaR) arxiv:2203.14465 — "hint-based
rationalization": inject the ground-truth answer as a hint, generate a
rationale that arrives at it, discard the hint at inference time.

Here the "answer" is the set of test outcomes (which tests pass/fail). We
inject them as a hint comment prepended to the problem prompt. The pipeline
runs again; if it now succeeds, we keep the artifacts as rationalized training
data.
"""

from __future__ import annotations

import logging
from typing import Callable

from corpus_producer.models import PhaseArtifact

logger = logging.getLogger(__name__)

MIN_EXAMPLES_PER_BIN = 60


def _build_hint_prompt(problem_prompt: str, test_hints: list[str]) -> str:
    """Prepend ground-truth test outcome hints to a problem prompt.

    Args:
        problem_prompt: Original problem text.
        test_hints: List of test outcome strings, e.g.
            ["test_add_two_numbers: PASS", "test_edge_case: FAIL"].

    Returns:
        Augmented prompt with hints block prepended.
    """
    hints_block = "\n".join(f"  # {h}" for h in test_hints)
    return (
        f"# Ground-truth test hints (use to guide your solution):\n"
        f"{hints_block}\n\n"
        f"{problem_prompt}"
    )


def star_rationalize(
    bin_key: str,
    existing_artifacts: list[PhaseArtifact],
    failing_problems: list[tuple[str, str, str]],  # (benchmark, problem_id, prompt)
    test_hints_by_problem: dict[str, list[str]],
    pipeline_runner: Callable[..., object],
    success_filter_fn: Callable[..., list[PhaseArtifact]],
    *,
    timeout: int = 300,
    base_model_id: str = "Qwen/Qwen3.5-9B",
) -> list[PhaseArtifact]:
    """Attempt rationalization on failing problems to pad a thin bin.

    Runs the pipeline with hint-augmented prompts for each failing problem.
    Keeps artifacts that pass the success filter, marks them rationalized=True.
    Stops once the bin reaches MIN_EXAMPLES_PER_BIN.

    Args:
        bin_key: The oracle bin that needs padding (for logging).
        existing_artifacts: Already-collected artifacts in this bin.
        failing_problems: List of (benchmark, problem_id, prompt) triples that
            failed in the primary pass.
        test_hints_by_problem: Map from problem_id to list of hint strings.
        pipeline_runner: Callable matching PipelineRunnerProtocol.
        success_filter_fn: Callable matching filter_artifacts signature.
        timeout: Per-run subprocess timeout.
        base_model_id: Base model for pipeline runs.

    Returns:
        New rationalized artifacts (not including existing_artifacts).
        Each returned artifact has rationalized=True.
    """
    from corpus_producer.pipeline_runner import PipelineRunResult

    new_artifacts: list[PhaseArtifact] = []
    current_count = len(existing_artifacts)

    for benchmark, problem_id, prompt in failing_problems:
        if current_count + len(new_artifacts) >= MIN_EXAMPLES_PER_BIN:
            logger.info(
                "Bin %s reached %d examples; stopping rationalization.",
                bin_key,
                MIN_EXAMPLES_PER_BIN,
            )
            break

        hints = test_hints_by_problem.get(problem_id, [])
        if not hints:
            logger.debug(
                "No hints for %s/%s; skipping rationalization.",
                benchmark,
                problem_id,
            )
            continue

        hint_prompt = _build_hint_prompt(prompt, hints)
        logger.info(
            "Rationalizing %s/%s for bin %s", benchmark, problem_id, bin_key
        )

        result: PipelineRunResult = pipeline_runner(  # type: ignore[assignment]
            benchmark,
            problem_id,
            hint_prompt,
            timeout=timeout,
            base_model_id=base_model_id,
        )

        if not result.success:
            logger.debug(
                "Rationalization pipeline failed for %s/%s", benchmark, problem_id
            )
            continue

        filtered = success_filter_fn(
            result.artifacts,
            result.final_code,
            benchmark,
            problem_id,
        )

        for art in filtered:
            art.rationalized = True
        new_artifacts.extend(filtered)

    logger.info(
        "Rationalization for bin %s produced %d new artifacts.",
        bin_key,
        len(new_artifacts),
    )
    return new_artifacts
