"""Config-driven benchmark runner for LLM evaluation across standardised datasets.

Provides a clean pipeline that loads a generation backend (vLLM in-process by
default, or any ``InferenceProvider`` from ``libs/inference/``), iterates over
each configured dataset benchmark, scores predictions against ground-truth, and
writes structured artefacts to:

    output_dir / run_id / dataset_name / model_name /
        ├── results.json   — per-problem results with full metadata
        ├── summary.json   — aggregate accuracy + throughput metrics
        └── config.json    — full run configuration snapshot

Quick-start (Python)::

    from evaluation.benchmark_runner import BenchmarkRunner
    from evaluation.config import load_config
    runner = BenchmarkRunner(load_config("configs/qwen3_5_olym_easy.yaml"))
    summaries = runner.run()

Provider selection
------------------
* ``ModelConfig(provider="vllm")`` with no ``base_url``  →  ``VLLMBackend``
  (``PyVLLMProvider`` in-process, no server required).
* ``ModelConfig(provider="vllm", base_url="http://…")``  →
  ``InferenceProviderBackend`` wrapping ``VLLMProvider`` (HTTP server must
  already be running).
* ``ModelConfig(provider="ollama|transformers|llamacpp")``  →
  ``InferenceProviderBackend`` via ``inference.factory.get_provider``.

Configuration and helper utilities are split into dedicated modules:
  - ``evaluation.config``  — dataclasses, registry, YAML loader
  - ``evaluation.utils``   — scoring, dataset loading, template rendering,
                             tool-call parsing, output helpers
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

from inference.benchmark_backends import (
    Backend,
    GenerationOutput,
    InferenceProviderBackend,
    VLLMBackend,
    _majority_vote,
    _tally_votes,
)

from evaluation.config import BenchmarkRunConfig, DatasetConfig, ModelConfig
from shared.mathbox import MAX_CODE_LINES, MATH_TOOLS, near_int_hint, parse_tool_calls

from evaluation.utils import (
    SCORERS,
    _clean_response,
    _error_result,
    _extract_answer,
    _resolve_system_prompt,
    _split_thinking,
    compute_summary,
    load_problems,
    save_results,
)

logger = logging.getLogger(__name__)

# ── Backend factory ───────────────────────────────────────────────────────────


def build_backend(cfg: ModelConfig) -> Backend:
    """Construct the appropriate ``Backend`` from ``ModelConfig``.

    ``provider="vllm"`` with no ``base_url``  →  ``VLLMBackend``
    (``PyVLLMProvider`` in-process, no server required).

    Anything else  →  ``InferenceProviderBackend`` via ``inference.factory``.
    """
    if cfg.provider == "vllm" and cfg.base_url is None:
        return VLLMBackend(cfg)

    from inference.factory import get_provider  # type: ignore[import-untyped]

    provider = get_provider(provider_type=cfg.provider, base_url=cfg.base_url)
    return InferenceProviderBackend(provider, cfg.model_id)


# ── Agentic loop (tool-calling + MathSandbox) ─────────────────────────────────


def _solve_batch_agentic(
    problems: list[dict[str, Any]],
    system_prompt: str,
    backend: Backend,
    model_cfg: ModelConfig,
    n_samples: int = 1,
) -> tuple[list[dict[str, Any]], float]:
    """Batched generate → tool-call → execute loop for a chunk of problems.

    Each ``(problem, sample)`` pair gets its own ``MathSandbox`` (stateful
    IPython kernel).  In every iteration the runner issues a single
    ``backend.generate_batch()`` call covering **all active trajectories**
    simultaneously — that is, all N problems × M samples that have not yet
    produced a final answer.  Tool calls are dispatched to each trajectory's
    own sandbox independently.

    When ``n_samples > 1`` the M completed trajectories per problem are
    aggregated via majority vote over their extracted ``\\boxed{}`` answers.
    The representative trajectory (whose answer matches the majority) is stored
    in ``messages`` / ``final_response``; all per-sample answers and full
    trajectories are preserved in ``all_answers``, ``vote_counts``, and
    ``all_trajectories``.

    Args:
        problems: Normalised problem dicts (with ``"id"``, ``"prompt"``).
        system_prompt: Rendered system prompt (already resolved).
        backend: Generation backend.  Tool calling requires ``VLLMBackend``.
        model_cfg: Model configuration (``max_iterations``, ``sandbox_timeout``,
            ``max_new_tokens``, ``temperature``, ``enable_thinking``).
        n_samples: Number of independent agentic trajectories per problem.
            Each trajectory runs in its own sandbox.  Majority vote is applied
            when > 1.  Defaults to 1 (single trajectory, original behaviour).

    Returns:
        ``(per_problem_results, total_wall_time_s)`` where each result dict
        contains ``model_answer``, ``final_response``, ``n_iterations``,
        ``n_gen_calls``, ``total_output_tokens``, ``total_gen_time_s``, and
        (when ``n_samples > 1``) ``all_answers``, ``vote_counts``,
        ``all_trajectories``.
    """
    import time as _time
    from contextlib import ExitStack

    from shared.mathbox import MathConfig, MathSandbox  # deferred GPU/kernel import

    sandbox_cfg = MathConfig(default_timeout=model_cfg.sandbox_timeout)
    n_probs = len(problems)
    total_trajectories = n_probs * n_samples

    # Flat list of N*M states — one per (problem, sample) pair.
    # flat_idx = prob_idx * n_samples + sample_idx maps into the sandboxes list.
    states: list[dict[str, Any]] = [
        {
            "prob_idx": i,
            "sample_idx": j,
            "flat_idx": i * n_samples + j,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problems[i]["prompt"]},
            ],
            "done": False,
            "model_answer": None,
            "final_response": "",
            "n_gen_calls": 0,
            "total_output_tokens": 0,
            "total_gen_time_s": 0.0,
            "n_iterations": 0,
            "last_results": [],
        }
        for i in range(n_probs)
        for j in range(n_samples)
    ]

    t_wall = _time.perf_counter()

    with ExitStack() as stack:
        # One independent sandbox per trajectory so kernel state never leaks
        # between samples of the same problem.
        sandboxes = [
            stack.enter_context(MathSandbox(sandbox_cfg))
            for _ in range(total_trajectories)
        ]

        for iteration in range(model_cfg.max_iterations):
            active = [s for s in states if not s["done"]]
            if not active:
                break

            logger.debug(
                "Agentic iteration %d — %d/%d trajectories still active",
                iteration + 1,
                len(active),
                total_trajectories,
            )

            # One batched call covering every active (problem, sample) trajectory
            outputs = backend.generate_batch(
                [s["messages"] for s in active],
                max_new_tokens=model_cfg.max_new_tokens,
                temperature=model_cfg.temperature,
                enable_thinking=model_cfg.enable_thinking,
                n_samples=1,        # 1 completion per agentic step; diversity
                tools=MATH_TOOLS,   # comes from independent trajectory histories
            )

            for state, out in zip(active, outputs):
                state["total_gen_time_s"] += out.elapsed_s
                state["total_output_tokens"] += out.n_tokens
                state["n_gen_calls"] += 1
                state["n_iterations"] = iteration + 1

                raw = out.text
                thinking, body = _split_thinking(raw)
                tool_calls = parse_tool_calls(body)

                if not tool_calls:
                    resp = _clean_response(body)
                    state["messages"].append({"role": "assistant", "content": resp})
                    state["model_answer"] = (
                        _extract_answer(resp)
                        or _extract_answer(thinking)
                        or _extract_answer(raw)
                    )
                    state["final_response"] = resp
                    state["done"] = True
                else:
                    state["messages"].append({"role": "assistant", "content": raw})
                    sandbox = sandboxes[state["flat_idx"]]
                    for tc in tool_calls:
                        code = tc["arguments"].get("code", "")
                        lines = code.splitlines()
                        if len(lines) > MAX_CODE_LINES:
                            result = (
                                f"[CODE TOO LONG — {len(lines)} lines, max {MAX_CODE_LINES}] "
                                "State your answer with \\boxed{}."
                            )
                        elif tc["name"] == "execute_python":
                            result = sandbox.execute(code)
                        else:
                            result = f"[ERROR] Unknown tool '{tc['name']}'."

                        hint = near_int_hint(result)
                        if hint:
                            result += hint

                        state["messages"].append({"role": "tool", "content": result})
                        state["last_results"].append(result)

                        # Stuck detection: same output 3× in a row → bail
                        if (
                            len(state["last_results"]) >= 3
                            and len(set(state["last_results"][-3:])) == 1
                        ):
                            bail = "[stuck] Same result 3× in a row."
                            state["messages"].append(
                                {"role": "assistant", "content": bail}
                            )
                            state["done"] = True
                            state["final_response"] = bail
                            break

    wall_time = _time.perf_counter() - t_wall

    # Aggregate flat states back into one result dict per problem.
    results: list[dict[str, Any]] = []
    for i in range(n_probs):
        prob_states = states[i * n_samples : (i + 1) * n_samples]

        for s in prob_states:
            if not s["done"]:
                s["final_response"] = "[max iterations reached]"

        all_answers = [s["model_answer"] for s in prob_states]
        total_tokens = sum(s["total_output_tokens"] for s in prob_states)
        # Samples run in the same batched forward passes, so wall time is the
        # max across the M trajectories rather than their sum.
        elapsed = max(s["total_gen_time_s"] for s in prob_states)

        if n_samples > 1:
            majority = _majority_vote(all_answers)
            vote_counts: dict[str, int] | None = _tally_votes(all_answers)
            rep = prob_states[0]
            if majority is not None:
                for s in prob_states:
                    if s["model_answer"] == majority:
                        rep = s
                        break
        else:
            majority = all_answers[0]
            vote_counts = None
            rep = prob_states[0]

        result: dict[str, Any] = {
            "model_answer": majority,
            "final_response": rep["final_response"],
            "messages": rep["messages"],
            "n_iterations": rep["n_iterations"],
            "n_gen_calls": rep["n_gen_calls"],
            "total_output_tokens": total_tokens,
            "total_gen_time_s": elapsed,
        }

        if n_samples > 1:
            result["all_answers"] = all_answers
            result["vote_counts"] = vote_counts
            result["all_trajectories"] = [
                {
                    "sample_idx": s["sample_idx"],
                    "model_answer": s["model_answer"],
                    "final_response": s["final_response"],
                    "messages": s["messages"],
                    "n_iterations": s["n_iterations"],
                    "n_gen_calls": s["n_gen_calls"],
                    "total_output_tokens": s["total_output_tokens"],
                }
                for s in prob_states
            ]

        results.append(result)

    return results, wall_time


# ── Inference loop ────────────────────────────────────────────────────────────


def run_dataset_benchmark(
    problems: list[dict[str, Any]],
    ds_cfg: DatasetConfig,
    backend: Backend,
    model_cfg: ModelConfig,
    system_prompt: str,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """Run inference for all problems in one dataset.

    When ``model_cfg.use_tools=True`` (default), delegates to
    ``_solve_batch_agentic`` which runs the full generate → tool-call →
    execute loop using ``MathSandbox``.  Otherwise falls back to simple
    single-pass batch inference with optional majority vote.

    Failed batches (either mode) are retried up to ``max_retries`` times
    before being recorded as error rows so the run continues cleanly.

    Returns:
        Per-problem result dicts.  Agentic mode adds ``n_iterations``,
        ``n_gen_calls``, and the full ``messages`` history.  Error rows
        carry ``error: true`` and ``error_reason: str``.
    """
    scorer = SCORERS.get(ds_cfg.scorer)
    if scorer is None:
        raise ValueError(
            f"Unknown scorer '{ds_cfg.scorer}'.  Available: {sorted(SCORERS)}"
        )

    results: list[dict[str, Any]] = []
    batch_size = model_cfg.batch_size
    n_batches = (len(problems) + batch_size - 1) // batch_size

    for chunk_idx, chunk_start in enumerate(range(0, len(problems), batch_size)):
        chunk = problems[chunk_start : chunk_start + batch_size]

        if model_cfg.use_tools:
            # ── Agentic tool-calling loop ──────────────────────────────────
            logger.debug(
                "Agentic batch %d/%d  dataset='%s'  rows=%d–%d",
                chunk_idx + 1,
                n_batches,
                ds_cfg.name,
                chunk_start + 1,
                chunk_start + len(chunk),
            )

            agentic_outputs: list[dict[str, Any]] | None = None
            last_error = ""
            for attempt in range(1, max_retries + 2):
                try:
                    agentic_outputs, _wall = _solve_batch_agentic(
                        chunk, system_prompt, backend, model_cfg,
                        n_samples=model_cfg.n_samples,
                    )
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    if attempt <= max_retries:
                        logger.warning(
                            "Agentic batch %d/%d attempt %d/%d failed (%s), retrying",
                            chunk_idx + 1, n_batches, attempt, max_retries + 1,
                            type(exc).__name__, exc_info=False,
                        )
                    else:
                        logger.error(
                            "Agentic batch %d/%d exhausted %d retries (%s) — recording errors",
                            chunk_idx + 1, n_batches, max_retries,
                            type(exc).__name__, exc_info=True,
                        )

            if agentic_outputs is None:
                results.extend(_error_result(p, error_reason=last_error) for p in chunk)
                continue

            for prob, ag in zip(chunk, agentic_outputs):
                model_answer = ag["model_answer"]
                correct = scorer(model_answer, prob["ground_truth"])
                tok_per_sec = (
                    ag["total_output_tokens"] / ag["total_gen_time_s"]
                    if ag["total_gen_time_s"] > 0 else 0.0
                )

                logger.debug(
                    "%s  %s  pred=%r  gt=%r  iters=%d  tokens=%d  %.2fs",
                    "PASS" if correct else "FAIL",
                    prob["id"],
                    model_answer,
                    prob["ground_truth"],
                    ag["n_iterations"],
                    ag["total_output_tokens"],
                    ag["total_gen_time_s"],
                )

                row: dict[str, Any] = {
                    "problem_id": prob["id"],
                    "prompt": prob["prompt"],
                    "ground_truth": prob["ground_truth"],
                    "model_answer": model_answer,
                    "correct": correct,
                    "n_iterations": ag["n_iterations"],
                    "n_gen_calls": ag["n_gen_calls"],
                    "n_tokens": ag["total_output_tokens"],
                    "elapsed_s": ag["total_gen_time_s"],
                    "tok_per_sec": tok_per_sec,
                    "final_response": ag["final_response"],
                    "messages": ag["messages"],
                }
                if model_cfg.n_samples > 1:
                    row["n_samples"] = model_cfg.n_samples
                    row["all_answers"] = ag.get("all_answers")
                    row["vote_counts"] = ag.get("vote_counts")
                    row["all_trajectories"] = ag.get("all_trajectories")
                results.append(row)

        else:
            # ── Simple single-pass batch inference ─────────────────────────
            n_samples = model_cfg.n_samples
            logger.debug(
                "Simple batch %d/%d  dataset='%s'  rows=%d–%d  n_samples=%d",
                chunk_idx + 1,
                n_batches,
                ds_cfg.name,
                chunk_start + 1,
                chunk_start + len(chunk),
                n_samples,
            )

            messages_list = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p["prompt"]},
                ]
                for p in chunk
            ]

            outputs: list[GenerationOutput] | None = None
            last_error = ""
            for attempt in range(1, max_retries + 2):
                try:
                    outputs = backend.generate_batch(
                        messages_list,
                        max_new_tokens=model_cfg.max_new_tokens,
                        temperature=model_cfg.temperature,
                        enable_thinking=model_cfg.enable_thinking,
                        n_samples=n_samples,
                    )
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    if attempt <= max_retries:
                        logger.warning(
                            "Batch %d/%d attempt %d/%d failed (%s), retrying",
                            chunk_idx + 1, n_batches, attempt, max_retries + 1,
                            type(exc).__name__, exc_info=False,
                        )
                    else:
                        logger.error(
                            "Batch %d/%d exhausted %d retries (%s) — recording errors",
                            chunk_idx + 1, n_batches, max_retries,
                            type(exc).__name__, exc_info=True,
                        )

            if outputs is None:
                results.extend(_error_result(p, error_reason=last_error) for p in chunk)
                continue

            for prob, out in zip(chunk, outputs):
                vote_counts: dict[str, int] | None = None
                if n_samples > 1 and out.all_texts is not None:
                    all_sample_answers = [_extract_answer(t) for t in out.all_texts]
                    model_answer = _majority_vote(all_sample_answers)
                    vote_counts = _tally_votes(all_sample_answers)
                else:
                    all_sample_answers = None
                    model_answer = _extract_answer(out.text)

                correct = scorer(model_answer, prob["ground_truth"])
                tok_per_sec = out.n_tokens / out.elapsed_s if out.elapsed_s > 0 else 0.0

                logger.debug(
                    "%s  %s  pred=%r  gt=%r  tokens=%d  %.2fs%s",
                    "PASS" if correct else "FAIL",
                    prob["id"],
                    model_answer,
                    prob["ground_truth"],
                    out.n_tokens,
                    out.elapsed_s,
                    f"  votes={vote_counts}" if vote_counts else "",
                )

                row = {
                    "problem_id": prob["id"],
                    "prompt": prob["prompt"],
                    "ground_truth": prob["ground_truth"],
                    "raw_output": out.text,
                    "model_answer": model_answer,
                    "correct": correct,
                    "n_tokens": out.n_tokens,
                    "elapsed_s": out.elapsed_s,
                    "tok_per_sec": tok_per_sec,
                }
                if n_samples > 1:
                    row["n_samples"] = n_samples
                    row["all_answers"] = all_sample_answers
                    row["vote_counts"] = vote_counts
                results.append(row)

    return results


# ── Runner ────────────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """Orchestrates a complete benchmark run across all configured datasets.

    Constructs the backend **once**, iterates over every dataset, runs
    inference, scores, and saves results.  The backend is torn down in a
    ``finally`` block so GPU memory is always released even if a dataset fails.

    Args:
        cfg: Complete run configuration (from ``load_config()`` or inline).

    Example::

        from evaluation.benchmark_runner import BenchmarkRunner
        from evaluation.config import load_config
        BenchmarkRunner(load_config("configs/qwen3_5_olym_easy.yaml")).run()
    """

    def __init__(self, cfg: BenchmarkRunConfig) -> None:
        self._cfg = cfg

    def run(self) -> list[dict[str, Any]]:
        """Execute all dataset benchmarks.

        Returns:
            Ordered list of summary dicts — one per successfully evaluated
            dataset.  Datasets that fail to load are skipped; their absence
            signals the failure.
        """
        cfg = self._cfg
        logger.info(
            "Benchmark run started  run_id=%s  model=%s  "
            "use_tools=%s  n_samples=%d  datasets=%s",
            cfg.run_id,
            cfg.model.model_id,
            cfg.model.use_tools,
            cfg.model.n_samples,
            [d.name for d in cfg.datasets],
        )

        backend = build_backend(cfg.model)
        all_summaries: list[dict[str, Any]] = []

        try:
            for ds_cfg in cfg.datasets:
                logger.info("── Evaluating: %s ──", ds_cfg.name)
                t_start = time.perf_counter()

                try:
                    problems = load_problems(ds_cfg)
                except Exception:
                    logger.exception(
                        "Failed to load dataset '%s', skipping", ds_cfg.name
                    )
                    continue

                system_prompt = _resolve_system_prompt(ds_cfg, cfg)

                try:
                    results = run_dataset_benchmark(
                        problems=problems,
                        ds_cfg=ds_cfg,
                        backend=backend,
                        model_cfg=cfg.model,
                        system_prompt=system_prompt,
                        max_retries=cfg.max_retries,
                    )
                except Exception:
                    logger.exception(
                        "Benchmark loop crashed for '%s', skipping", ds_cfg.name
                    )
                    continue

                elapsed = time.perf_counter() - t_start
                summary = compute_summary(
                    results=results,
                    dataset_name=ds_cfg.name,
                    model_id=cfg.model.model_id,
                    run_id=cfg.run_id,
                    elapsed_s=elapsed,
                    n_samples=cfg.model.n_samples,
                )

                save_results(results, summary, cfg, ds_cfg.name)
                all_summaries.append(summary)

                logger.info(
                    "Completed '%s': %d/%d correct (%.1f%%)  "
                    "errors=%d  avg_tok/s=%.1f  wall=%.1fs",
                    ds_cfg.name,
                    summary["correct"],
                    summary["total"],
                    summary["accuracy"] * 100,
                    summary["errors"],
                    summary["avg_tok_per_sec"],
                    elapsed,
                )

        finally:
            backend.close()

        logger.info(
            "Run %s complete — %d/%d dataset(s) evaluated",
            cfg.run_id,
            len(all_summaries),
            len(cfg.datasets),
        )
        return all_summaries
