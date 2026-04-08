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

import json
import logging
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, IO

from inference.benchmark_backends import (
    Backend,
    GenerationOutput,
    InferenceProviderBackend,
    VLLMBackend,
    _majority_vote,
    _tally_votes,
)

from evaluation.config import BenchmarkRunConfig, DatasetConfig, ModelConfig, RetrieverConfig
from shared.mathbox import MAX_CODE_LINES, MATH_TOOLS, near_int_hint, parse_tool_calls

from evaluation.utils import (
    SCORERS,
    _clean_response,
    _dataset_slug,
    _error_result,
    _extract_answer,
    _extract_boxed,
    _model_slug,
    _resolve_system_prompt,
    _split_thinking,
    compute_summary,
    load_problems,
    render_template,
    save_results,
)

logger = logging.getLogger(__name__)

# ── Backend factory ───────────────────────────────────────────────────────────


def build_backend(cfg: ModelConfig) -> Backend:
    """Construct the appropriate ``Backend`` from ``ModelConfig``.

    * ``provider="vllm"`` with no ``base_url``  →  ``VLLMBackend``
      (``PyVLLMProvider`` in-process, no server required).
    * ``provider="vllm_server"``  →  ``VLLMServerBackend`` (OpenAI client
      against a running ``vllm serve`` process; tool calls are structured).
    * ``provider="vllm"`` **with** ``base_url``  →  also ``VLLMServerBackend``
      (convenience: same config file works for in-process and server mode by
      toggling ``base_url``).
    * ``provider="hf_router"``  →  ``HFRouterBackend`` (HF Inference Router;
      requires ``HF_TOKEN`` env var).
    * Anything else  →  ``InferenceProviderBackend`` via ``inference.factory``
      (no tool support — for simple non-agentic providers).
    """
    from inference.benchmark_backends import HFRouterBackend, VLLMServerBackend

    if cfg.provider == "vllm" and cfg.base_url is None:
        return VLLMBackend(cfg)

    if cfg.provider in ("vllm_server",) or (
        cfg.provider == "vllm" and cfg.base_url is not None
    ):
        return VLLMServerBackend(cfg)

    if cfg.provider == "hf_router":
        return HFRouterBackend(cfg)

    from inference.factory import get_provider  # type: ignore[import-untyped]

    provider = get_provider(provider_type=cfg.provider, base_url=cfg.base_url)
    return InferenceProviderBackend(provider, cfg.model_id)


# ── Context retriever helpers ─────────────────────────────────────────────────


def _build_retriever(
    ret_cfg: RetrieverConfig,
    eval_dataset_configs: list[DatasetConfig],
) -> Any:
    """Construct and index a ``MathContextRetriever`` from a ``RetrieverConfig``.

    The index is built once and cached on disk; subsequent calls with the same
    configuration load in seconds.  ``dedup_test_path=null`` in the YAML auto-
    detects the test-set path from the configured evaluation datasets so that
    train examples identical to test problems are excluded.

    Args:
        ret_cfg: Retriever section parsed from the YAML config.
        eval_dataset_configs: The benchmark ``DatasetConfig`` list — used to
            auto-detect the dedup test-set path when ``ret_cfg.dedup_test_path``
            is ``None``.

    Returns:
        A ready ``MathContextRetriever`` with its index built/loaded.
    """
    from pathlib import Path as _Path

    from evaluation.config import DATASET_REGISTRY, _DATA_ROOT
    from shared.math_retriever import (  # noqa: PLC0415
        DatasetConfig as MathDSCfg,
        MathContextRetriever,
        MathRetrievalConfig,
    )

    # Convert per-dataset config dicts → MathDSCfg objects.
    _known_ds = set(MathDSCfg.__dataclass_fields__)  # type: ignore[attr-defined]
    math_datasets: dict[str, MathDSCfg] = {}
    for ds_name, raw in ret_cfg.datasets.items():
        if isinstance(raw, dict):
            math_datasets[ds_name] = MathDSCfg(
                **{k: v for k, v in raw.items() if k in _known_ds}
            )
        else:
            math_datasets[ds_name] = MathDSCfg(n=int(raw))

    # Resolve dedup path: explicit > auto-detect from eval datasets > None.
    dedup_path: _Path | None = None
    if ret_cfg.dedup_test_path:
        dedup_path = _Path(ret_cfg.dedup_test_path)
    else:
        for ds_cfg in eval_dataset_configs:
            registry_rel = DATASET_REGISTRY.get(ds_cfg.name)
            candidate = (
                _Path(ds_cfg.data_path)
                if ds_cfg.data_path
                else (_DATA_ROOT / registry_rel if registry_rel else None)
            )
            if candidate and candidate.exists():
                dedup_path = candidate
                logger.info(
                    "Retriever dedup: auto-detected test path %s from dataset '%s'",
                    dedup_path,
                    ds_cfg.name,
                )
                break

    index_dir = (
        _Path(ret_cfg.index_dir)
        if ret_cfg.index_dir
        else _Path.home() / ".rune" / "math_index"
    )

    math_cfg = MathRetrievalConfig(
        datasets=math_datasets,
        embedding_model=ret_cfg.embedding_model,
        top_k=ret_cfg.top_k,
        tir_top_k=ret_cfg.tir_top_k,
        max_solution_chars=ret_cfg.max_solution_chars,
        similarity_threshold=ret_cfg.similarity_threshold,
        dedup_test_path=dedup_path,
        index_dir=index_dir,
        data_root=_DATA_ROOT,
    )

    retriever = MathContextRetriever(math_cfg)
    n = retriever.build_index()
    logger.info("Retriever ready — %d examples indexed", n)
    return retriever


def _tokens_approx(text: str) -> int:
    """Approximate token count using the 1 token ≈ 4 characters heuristic."""
    return max(1, len(text) // 4)


def _inject_retrieved_examples(
    problem_text: str,
    retriever: Any,
    ds_cfg: DatasetConfig,
    run_cfg: BenchmarkRunConfig,
) -> tuple[str, list[Any]]:
    """Render the system prompt with retrieved few-shot examples injected.

    Calls the retriever, formats the examples, then re-renders the active
    Jinja2 template with ``retrieved_examples`` added to the template vars.
    Falls back to the base system prompt if no template is configured.

    Args:
        problem_text: The raw problem string used as the retrieval query.
        retriever: A built ``MathContextRetriever`` instance.
        ds_cfg: Dataset-level config (carries template name and vars).
        run_cfg: Run-level config (carries template name, vars, and dir).

    Returns:
        ``(rendered_system_prompt, raw_examples)`` where ``raw_examples`` is
        the list of ``MathExample`` objects returned by the retriever (used by
        :func:`_save_retrieval_context` to persist the retrieval log).
    """
    examples = retriever.query(problem_text)
    retrieved_examples = retriever.format_context(examples)

    template = ds_cfg.system_prompt_template or run_cfg.system_prompt_template
    if not template:
        # No template active — return base prompt unmodified.
        prompt = ds_cfg.system_prompt or run_cfg.system_prompt or ""
        return prompt, examples

    base_vars: dict[str, Any] = {**run_cfg.template_vars}
    if ds_cfg.system_prompt_template:
        base_vars.update(ds_cfg.template_vars)
    base_vars["retrieved_examples"] = retrieved_examples

    return render_template(template, base_vars, run_cfg.templates_dir), examples


def _save_retrieval_context(
    problems: list[dict[str, Any]],
    per_problem_examples: list[list[Any]],
    out_dir: Path,
) -> None:
    """Persist per-problem retrieval data to ``out_dir/retrieval_context.json``.

    Writes a JSON array — one object per problem — recording the problem ID,
    approximate token counts, and the full ranked list of retrieved examples
    with their similarity scores, sources, and metadata.  The file is written
    once, immediately after the retrieval pre-computation step, so it is
    available even if the generation loop is interrupted.

    Args:
        problems: Normalised problem dicts (must contain ``"id"`` and
            ``"prompt"``).
        per_problem_examples: Parallel list of ``MathExample`` lists returned
            by the retriever — one inner list per problem.
        out_dir: Directory where ``retrieval_context.json`` is written.
            Created if it does not already exist.
    """
    rows: list[dict[str, Any]] = []
    for prob, examples in zip(problems, per_problem_examples):
        retrieved: list[dict[str, Any]] = []
        for rank, ex in enumerate(examples, 1):
            p_tok = _tokens_approx(ex.problem)
            s_tok = _tokens_approx(ex.solution)
            retrieved.append(
                {
                    "rank": rank,
                    "source": ex.source,
                    "similarity": round(ex.similarity, 6),
                    "metadata": ex.metadata,
                    "problem": ex.problem,
                    "solution": ex.solution,
                    "problem_tokens": p_tok,
                    "solution_tokens": s_tok,
                    "total_tokens": p_tok + s_tok,
                }
            )
        rows.append(
            {
                "problem_id": prob["id"],
                "query_text": prob["prompt"],
                "query_tokens": _tokens_approx(prob["prompt"]),
                "n_retrieved": len(retrieved),
                "total_retrieved_tokens": sum(r["total_tokens"] for r in retrieved),
                "retrieved": retrieved,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_context.json"
    out_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Retrieval context saved → %s", out_path)


# ── Agentic loop (tool-calling + MathSandbox) ─────────────────────────────────


def _solve_batch_agentic(
    problems: list[dict[str, Any]],
    system_prompt: str,
    backend: Backend,
    model_cfg: ModelConfig,
    n_samples: int = 1,
    system_prompts: list[str] | None = None,
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
        system_prompt: Rendered system prompt shared across all problems.
            Ignored for any position where *system_prompts* provides a value.
        backend: Generation backend.  Tool calling requires ``VLLMBackend``.
        model_cfg: Model configuration (``max_iterations``, ``sandbox_timeout``,
            ``max_new_tokens``, ``temperature``, ``enable_thinking``).
        n_samples: Number of independent agentic trajectories per problem.
            Each trajectory runs in its own sandbox.  Majority vote is applied
            when > 1.  Defaults to 1 (single trajectory, original behaviour).
        system_prompts: Optional per-problem system prompts.  When provided
            (length must equal ``len(problems)``), each problem uses its own
            rendered prompt — enabling per-problem retrieval context injection.
            Falls back to *system_prompt* for any missing position.

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
                {
                    "role": "system",
                    "content": (
                        system_prompts[i]
                        if system_prompts and i < len(system_prompts)
                        else system_prompt
                    ),
                },
                {"role": "user", "content": problems[i]["prompt"]},
            ],
            "done": False,
            "model_answer": None,
            "candidate_answer": None,   # best boxed answer seen in tool-call iterations
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

                # Normalise tool calls into [{"id": str|None, "name": str,
                # "arguments": dict}].  Prefer structured data from OpenAI-
                # compatible backends (HFRouterBackend); fall back to parsing
                # the Qwen XML format embedded in the text (VLLMBackend).
                if out.tool_calls is not None:
                    norm_tcs: list[dict[str, Any]] = out.tool_calls
                else:
                    norm_tcs = [
                        {"id": None, "name": tc["name"], "arguments": tc["arguments"]}
                        for tc in parse_tool_calls(body)
                    ]

                if not norm_tcs:
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
                    # Build the assistant message.  OpenAI-compatible APIs
                    # require the tool_calls list in the message so subsequent
                    # turns can reference call IDs.  vLLM in-process just needs
                    # the raw text (the chat template handles the rest).
                    if out.tool_calls is not None:
                        state["messages"].append({
                            "role": "assistant",
                            "content": raw or None,
                            "tool_calls": [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["arguments"]),
                                    },
                                }
                                for tc in norm_tcs
                            ],
                        })
                    else:
                        state["messages"].append({"role": "assistant", "content": raw})

                    # Capture any boxed answer already present in this response
                    # as a candidate fallback.  The model sometimes states its
                    # answer *and* calls a tool in the same turn.
                    _interim_body = _clean_response(body)
                    _interim = (
                        _extract_boxed(_interim_body)
                        or _extract_boxed(thinking)
                        or _extract_boxed(raw)
                    )
                    if _interim is not None:
                        state["candidate_answer"] = _interim

                    sandbox = sandboxes[state["flat_idx"]]
                    for tc in norm_tcs:
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

                        # OpenAI protocol: tool result messages must carry the
                        # matching tool_call_id so the API can pair them up.
                        tool_msg: dict[str, Any] = {"role": "tool", "content": result}
                        if tc.get("id"):
                            tool_msg["tool_call_id"] = tc["id"]
                        state["messages"].append(tool_msg)
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
            # When the loop ended without a clean final answer (stuck detection
            # fired, max iterations hit, or the final response had no boxed
            # expression), fall back to the best intermediate candidate seen
            # during tool-call iterations.
            if s["model_answer"] is None and s["candidate_answer"] is not None:
                s["model_answer"] = s["candidate_answer"]

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


def _fmt_eta(seconds: float) -> str:
    """Format a duration in seconds as a human-readable ETA string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def run_dataset_benchmark(
    problems: list[dict[str, Any]],
    ds_cfg: DatasetConfig,
    backend: Backend,
    model_cfg: ModelConfig,
    system_prompt: str,
    run_cfg: BenchmarkRunConfig | None = None,
    retriever: Any | None = None,
    max_retries: int = 2,
    out_dir: Path | None = None,
    run_id: str = "",
) -> list[dict[str, Any]]:
    """Run inference for all problems in one dataset.

    When ``model_cfg.use_tools=True`` (default), delegates to
    ``_solve_batch_agentic`` which runs the full generate → tool-call →
    execute loop using ``MathSandbox``.  Otherwise falls back to simple
    single-pass batch inference with optional majority vote.

    When *retriever* is supplied, per-problem system prompts are pre-computed
    by querying the retriever with each problem text and injecting the
    formatted few-shot block into the system-prompt template.  The base
    *system_prompt* is used as a fallback for any problem that produces no
    retrieved examples.

    Failed batches (either mode) are retried up to ``max_retries`` times
    before being recorded as error rows so the run continues cleanly.

    When ``out_dir`` is provided, results are streamed to
    ``out_dir/results.jsonl`` (one JSON object per line) as each problem
    completes, and ``out_dir/summary.json`` is refreshed with running
    totals after every row.  The caller is responsible for writing the
    final ``results.json`` and ``config.json`` (via ``save_results``).

    Args:
        problems: Normalised problem dicts (``id``, ``prompt``, ``ground_truth``).
        ds_cfg: Dataset-level configuration.
        backend: Generation backend.
        model_cfg: Model hyper-parameters and loop settings.
        system_prompt: Pre-rendered base system prompt (used when no retriever
            is active, or as a fallback).
        run_cfg: Full run configuration, required when *retriever* is supplied
            so the system-prompt template can be re-rendered with context.
        retriever: Optional ``MathContextRetriever`` instance.  When provided,
            per-problem prompts are built before the generation loop.
        max_retries: Retry budget per batch.
        out_dir: Directory for streamed JSONL + rolling summary output.
        run_id: Run identifier embedded in streaming summaries.

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

    # Pre-compute per-problem system prompts when the retriever is active.
    # This happens once upfront so the generation loop stays unmodified.
    per_problem_prompts: list[str] | None = None
    if retriever is not None and run_cfg is not None:
        logger.info(
            "Pre-computing retrieval context for %d problems …", len(problems)
        )
        _pairs = [
            _inject_retrieved_examples(p["prompt"], retriever, ds_cfg, run_cfg)
            for p in problems
        ]
        per_problem_prompts = [pair[0] for pair in _pairs]
        per_problem_examples = [pair[1] for pair in _pairs]

        if out_dir is not None:
            _save_retrieval_context(problems, per_problem_examples, out_dir)

        logger.info("Retrieval context ready")

    results: list[dict[str, Any]] = []
    batch_size = model_cfg.batch_size
    n_total = len(problems)
    n_batches = (n_total + batch_size - 1) // batch_size

    # Running progress counters
    n_done = 0
    n_correct = 0
    t_run_start = time.perf_counter()

    # Open JSONL stream if an output directory was provided
    jsonl_fh: IO[str] | None = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "results.jsonl"
        jsonl_fh = open(jsonl_path, "w", encoding="utf-8")  # noqa: SIM115
        logger.info("Streaming results → %s", jsonl_path)

    def _append_result(row: dict[str, Any]) -> None:
        """Append one result row to the JSONL file and refresh summary.json."""
        if out_dir is None or jsonl_fh is None:
            return
        jsonl_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_fh.flush()
        partial = compute_summary(
            results=results,
            dataset_name=ds_cfg.name,
            model_id=model_cfg.model_id,
            run_id=run_id,
            elapsed_s=time.perf_counter() - t_run_start,
            n_samples=model_cfg.n_samples,
        )
        partial["status"] = "in_progress"
        (out_dir / "summary.json").write_text(
            json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _log_progress(
        prob_id: str,
        correct: bool,
        elapsed_s: float,
        extra: str = "",
    ) -> None:
        nonlocal n_done, n_correct
        n_done += 1
        if correct:
            n_correct += 1

        elapsed_total = time.perf_counter() - t_run_start
        avg_s = elapsed_total / n_done
        remaining = n_total - n_done
        eta_str = f"  ETA ~{_fmt_eta(avg_s * remaining)}" if remaining > 0 else ""

        logger.info(
            "[%d/%d]  %s  %s  %.1fs  ──  %d/%d correct (%.1f%%)%s%s",
            n_done,
            n_total,
            "PASS" if correct else "FAIL",
            prob_id,
            elapsed_s,
            n_correct,
            n_done,
            100.0 * n_correct / n_done,
            eta_str,
            f"  {extra}" if extra else "",
        )

    for chunk_idx, chunk_start in enumerate(range(0, n_total, batch_size)):
        chunk = problems[chunk_start : chunk_start + batch_size]

        if model_cfg.use_tools:
            # ── Agentic tool-calling loop ──────────────────────────────────
            logger.info(
                "Batch %d/%d  (rows %d–%d)  dataset='%s'",
                chunk_idx + 1,
                n_batches,
                chunk_start + 1,
                chunk_start + len(chunk),
                ds_cfg.name,
            )

            chunk_prompts = (
                per_problem_prompts[chunk_start : chunk_start + len(chunk)]
                if per_problem_prompts is not None
                else None
            )

            agentic_outputs: list[dict[str, Any]] | None = None
            last_error = ""
            for attempt in range(1, max_retries + 2):
                try:
                    agentic_outputs, _wall = _solve_batch_agentic(
                        chunk, system_prompt, backend, model_cfg,
                        n_samples=model_cfg.n_samples,
                        system_prompts=chunk_prompts,
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
                for p in chunk:
                    err_row = _error_result(p, error_reason=last_error)
                    results.append(err_row)
                    _log_progress(p["id"], correct=False, elapsed_s=0.0, extra="ERROR")
                    _append_result(err_row)
                continue

            for prob, ag in zip(chunk, agentic_outputs):
                model_answer = ag["model_answer"]
                correct = scorer(model_answer, prob["ground_truth"])
                tok_per_sec = (
                    ag["total_output_tokens"] / ag["total_gen_time_s"]
                    if ag["total_gen_time_s"] > 0 else 0.0
                )

                _log_progress(
                    prob["id"],
                    correct=correct,
                    elapsed_s=ag["total_gen_time_s"],
                    extra=(
                        f"pred={model_answer!r}  gt={prob['ground_truth']!r}"
                        f"  iters={ag['n_iterations']}  tok={ag['total_output_tokens']}"
                        f"  {tok_per_sec:.0f}tok/s"
                    ),
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
                _append_result(row)

        else:
            # ── Simple single-pass batch inference ─────────────────────────
            n_samples = model_cfg.n_samples
            logger.info(
                "Batch %d/%d  (rows %d–%d)  dataset='%s'  n_samples=%d",
                chunk_idx + 1,
                n_batches,
                chunk_start + 1,
                chunk_start + len(chunk),
                ds_cfg.name,
                n_samples,
            )

            chunk_prompts = (
                per_problem_prompts[chunk_start : chunk_start + len(chunk)]
                if per_problem_prompts is not None
                else None
            )
            messages_list = [
                [
                    {
                        "role": "system",
                        "content": (
                            chunk_prompts[pi]
                            if chunk_prompts is not None
                            else system_prompt
                        ),
                    },
                    {"role": "user", "content": p["prompt"]},
                ]
                for pi, p in enumerate(chunk)
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
                for p in chunk:
                    err_row = _error_result(p, error_reason=last_error)
                    results.append(err_row)
                    _log_progress(p["id"], correct=False, elapsed_s=0.0, extra="ERROR")
                    _append_result(err_row)
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

                _log_progress(
                    prob["id"],
                    correct=correct,
                    elapsed_s=out.elapsed_s,
                    extra=(
                        f"pred={model_answer!r}  gt={prob['ground_truth']!r}"
                        f"  tok={out.n_tokens}  {tok_per_sec:.0f}tok/s"
                        + (f"  votes={vote_counts}" if vote_counts else "")
                    ),
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
                _append_result(row)

    if jsonl_fh is not None:
        jsonl_fh.close()

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
        self._retriever: Any | None = None
        if cfg.retriever is not None and cfg.retriever.use_retriever:
            logger.info("Building math context retriever …")
            self._retriever = _build_retriever(cfg.retriever, cfg.datasets)

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

                # Resolve output directory and write config snapshot upfront so
                # partial results are always paired with their configuration.
                out_dir = (
                    cfg.output_dir
                    / cfg.run_id
                    / _dataset_slug(ds_cfg.name)
                    / _model_slug(cfg.model.model_id)
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "config.json").write_text(
                    json.dumps(asdict(cfg), indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )

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
                        run_cfg=cfg,
                        retriever=self._retriever,
                        max_retries=cfg.max_retries,
                        out_dir=out_dir,
                        run_id=cfg.run_id,
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
