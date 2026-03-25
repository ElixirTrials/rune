"""Synchronous Backend abstraction for benchmark evaluation.

Bridges the async ``InferenceProvider`` API into a synchronous, batched
interface suited for benchmark runners.  Two concrete backends are provided:

- ``VLLMBackend``: in-process vLLM via ``PyVLLMProvider``.  Supports tool
  schemas, ``enable_thinking``, and batched N-sample generation using vLLM's
  native ``SamplingParams(n=N)`` so all samples are produced in a single
  forward-pass sweep.
- ``InferenceProviderBackend``: wraps any async ``InferenceProvider`` from
  ``libs/inference/``, bridging it with ``asyncio.run()`` / ``asyncio.gather``
  for concurrent N-sample requests.

Majority-vote helpers (``_majority_vote``, ``_tally_votes``) are exported here
as a single import point for benchmark runners.

**Design note — answer extraction stays in the benchmark runner.**
``GenerationOutput`` carries raw completion strings (``all_texts``) rather than
pre-extracted answers.  Scoring logic (``\\boxed{}`` extraction, numeric
comparison) belongs in the evaluation layer, not the inference layer.

Example::

    from inference.benchmark_backends import VLLMBackend, GenerationOutput

    backend = VLLMBackend(model_cfg)
    outs = backend.generate_batch(
        [messages1, messages2],
        max_new_tokens=1024,
        temperature=0.6,
    )
    backend.close()
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from inference.provider import InferenceProvider

logger = logging.getLogger(__name__)


# ── Output type ───────────────────────────────────────────────────────────────


@dataclass
class GenerationOutput:
    """Result of one (possibly multi-sample) generation call.

    Attributes:
        text: The output text.  For single-sample calls this is the only
            completion.  For multi-sample calls this is the *first* raw
            completion; callers should use ``all_texts`` for majority vote.
        n_tokens: Total output tokens summed across *all* samples.
        elapsed_s: Per-problem wall-clock time.  For batched calls this is
            ``total_wall / batch_size``; for multi-sample calls it is the
            full elapsed time for all N samples of that problem.
        all_texts: All N raw completion strings when ``n_samples > 1``.
            ``None`` for single-sample calls.  Answer extraction and majority
            voting are left to the caller so the inference layer stays
            task-agnostic.
    """

    text: str
    n_tokens: int
    elapsed_s: float
    all_texts: list[str] | None = None


# ── Voting helpers ────────────────────────────────────────────────────────────


def _majority_vote(answers: list[str | None]) -> str | None:
    """Return the most frequent non-``None`` answer, or ``None`` if all absent.

    Args:
        answers: Extracted answer strings (or ``None`` for failed extractions).

    Returns:
        The plurality answer, or ``None`` when no valid answer exists.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return max(set(valid), key=valid.count)


def _tally_votes(answers: list[str | None]) -> dict[str, int]:
    """Count occurrences of each non-``None`` answer.

    Args:
        answers: Extracted answer strings (or ``None`` for failed extractions).

    Returns:
        Mapping from answer string to its occurrence count.
    """
    counts: dict[str, int] = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    return counts


# ── Backend abstraction ───────────────────────────────────────────────────────


class Backend(ABC):
    """Synchronous generation backend for benchmark evaluation.

    Both methods accept OpenAI-style message lists; backends handle prompt
    formatting internally.  The ``n_samples`` parameter enables majority-vote
    evaluation without requiring callers to manage sampling loops.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool = False,
        n_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        """Generate one completion (or N completions for majority vote).

        Args:
            messages: OpenAI-style chat messages.
            max_new_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            enable_thinking: Enable Qwen3-style ``<think>…</think>`` prefix.
            n_samples: Number of independent samples.  When ``> 1``, the
                backend generates all samples in one call and returns them
                in ``GenerationOutput.all_texts``.
            tools: Optional tool-schema list forwarded to the chat template.
                Only supported by ``VLLMBackend``; ``InferenceProviderBackend``
                will log a warning and ignore it.
        """
        ...

    @abstractmethod
    def generate_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool = False,
        n_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[GenerationOutput]:
        """Batched equivalent of ``generate``.

        Wall-clock time is divided evenly across the batch so that
        ``GenerationOutput.elapsed_s`` represents effective per-problem
        latency rather than total batch wall time.
        """
        ...

    def close(self) -> None:
        """Release GPU memory or HTTP connections."""


# ── VLLMBackend ───────────────────────────────────────────────────────────────


class VLLMBackend(Backend):
    """Direct in-process vLLM backend backed by ``PyVLLMProvider``.

    For ``n_samples == 1`` uses a standard single-output ``SamplingParams``.
    For ``n_samples > 1`` passes ``n=n_samples`` to vLLM so that all samples
    for a prompt are generated in one forward-pass sweep, with raw completion
    strings returned in ``GenerationOutput.all_texts``.

    Args:
        cfg: A model configuration object with attributes:
            ``model_id``, ``tensor_parallel_size``, ``max_model_len``,
            ``dtype``, ``enable_prefix_caching``, ``language_model_only``.

    Example::

        backend = VLLMBackend(model_cfg)
        out = backend.generate(messages, max_new_tokens=1024, temperature=0.6)
        print(out.text)
        backend.close()
    """

    def __init__(self, cfg: Any) -> None:
        from inference.vllm_provider import PyVLLMProvider

        logger.info(
            "Initialising VLLMBackend: %s  (tp=%d  dtype=%s)",
            cfg.model_id,
            cfg.tensor_parallel_size,
            cfg.dtype,
        )
        self._cfg = cfg
        self._provider = PyVLLMProvider(
            model_id=cfg.model_id,
            tensor_parallel_size=cfg.tensor_parallel_size,
            max_model_len=cfg.max_model_len,
            dtype=cfg.dtype,
            enable_prefix_caching=cfg.enable_prefix_caching,
            language_model_only=cfg.language_model_only,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool = False,
        n_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        return self.generate_batch(
            [messages], max_new_tokens, temperature, enable_thinking, n_samples, tools
        )[0]

    def generate_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool = False,
        n_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[GenerationOutput]:
        from vllm import SamplingParams  # type: ignore[import-untyped]

        prompts = [
            self._provider.format_prompt(msgs, enable_thinking, tools)
            for msgs in messages_list
        ]

        # Enforce temperature diversity when sampling multiple independent times
        effective_temp = max(temperature, 0.4) if n_samples > 1 else max(temperature, 1e-6)
        sp = SamplingParams(
            n=n_samples,
            max_tokens=max_new_tokens,
            temperature=effective_temp,
            stop_token_ids=self._provider.stop_token_ids(),
        )

        t0 = time.perf_counter()
        raw_outs = self._provider.generate_batch_raw(prompts, sp)
        elapsed = time.perf_counter() - t0
        elapsed_each = elapsed / len(prompts)

        results: list[GenerationOutput] = []
        for out in raw_outs:
            completions = out.outputs
            total_toks = sum(len(c.token_ids) for c in completions)

            if n_samples == 1:
                results.append(
                    GenerationOutput(
                        text=completions[0].text,
                        n_tokens=total_toks,
                        elapsed_s=elapsed_each,
                    )
                )
            else:
                all_texts = [c.text for c in completions]
                results.append(
                    GenerationOutput(
                        text=all_texts[0],
                        n_tokens=total_toks,
                        elapsed_s=elapsed_each,
                        all_texts=all_texts,
                    )
                )

        return results

    def close(self) -> None:
        self._provider.close()


# ── InferenceProviderBackend ──────────────────────────────────────────────────


class InferenceProviderBackend(Backend):
    """Backend wrapping an async ``InferenceProvider`` from ``libs/inference/``.

    Uses ``asyncio.run()`` to bridge the async provider API into the
    synchronous ``Backend`` interface.  For ``n_samples > 1``, all samples
    for all problems are issued concurrently via ``asyncio.gather``.

    Tool schemas are not supported — if ``tools`` is provided a warning is
    logged and the parameter is silently ignored.  Use ``VLLMBackend`` for
    agentic tool-calling.

    Args:
        provider: A concrete ``InferenceProvider`` instance.
        model_id: Model identifier forwarded to ``provider.generate()``.

    Example::

        from inference.factory import get_provider
        provider = get_provider("ollama", base_url="http://localhost:11434/v1")
        backend = InferenceProviderBackend(provider, "llama3")
        out = backend.generate(messages, max_new_tokens=512, temperature=0.7)
    """

    def __init__(self, provider: InferenceProvider, model_id: str) -> None:
        self._provider = provider
        self._model_id = model_id

    @staticmethod
    def _msg_to_args(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return system, user

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool = False,
        n_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerationOutput:
        return self.generate_batch(
            [messages], max_new_tokens, temperature, enable_thinking, n_samples, tools
        )[0]

    def generate_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool = False,
        n_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[GenerationOutput]:
        if tools is not None:
            logger.warning(
                "InferenceProviderBackend does not support in-prompt tool schemas; "
                "tools parameter ignored.  Use VLLMBackend (provider='vllm', no "
                "base_url) for agentic tool-calling."
            )
        effective_temp = max(temperature, 0.4) if n_samples > 1 else temperature

        async def _gather() -> list[GenerationOutput]:
            tasks = []
            for messages in messages_list:
                system, user = self._msg_to_args(messages)
                for _ in range(n_samples):
                    tasks.append(
                        self._provider.generate(
                            prompt=user,
                            model=self._model_id,
                            system_prompt=system,
                            max_tokens=max_new_tokens,
                            temperature=effective_temp,
                        )
                    )

            t0 = time.perf_counter()
            flat_results = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0
            elapsed_each = elapsed / len(messages_list)

            outputs: list[GenerationOutput] = []
            for prob_idx in range(len(messages_list)):
                chunk = flat_results[prob_idx * n_samples : (prob_idx + 1) * n_samples]
                total_toks = sum(r.token_count for r in chunk)

                if n_samples == 1:
                    outputs.append(
                        GenerationOutput(
                            text=chunk[0].text,
                            n_tokens=total_toks,
                            elapsed_s=elapsed_each,
                        )
                    )
                else:
                    all_texts = [r.text for r in chunk]
                    outputs.append(
                        GenerationOutput(
                            text=all_texts[0],
                            n_tokens=total_toks,
                            elapsed_s=elapsed_each,
                            all_texts=all_texts,
                        )
                    )

            return outputs

        return asyncio.run(_gather())
