"""vLLM inference providers — server-backed and in-process.

Two concrete ``InferenceProvider`` implementations for vLLM:

``VLLMProvider`` (HTTP server)
    Communicates with a running vLLM server via its OpenAI-compatible REST API
    and vLLM's proprietary LoRA management endpoints.  Supports hot-loading and
    unloading LoRA adapters at runtime.  The server process is managed
    externally.

``PyVLLMProvider`` (in-process)
    Loads the model directly into the current process via ``vllm.LLM``.  No
    server required.  Owns the GPU memory; call ``close()`` to release it.
    LoRA hot-loading is not supported — use ``VLLMProvider`` for LoRA workflows.
    Exposes extra helpers (``format_prompt``, ``generate_batch_raw``,
    ``stop_token_ids``) used by ``inference.benchmark_backends.VLLMBackend``.

Example::

    # HTTP server (LoRA-capable)
    provider = VLLMProvider(base_url="http://localhost:8100/v1")
    result = await provider.generate("def fib", model="Qwen2.5-Coder-7B")

    # In-process (benchmark / offline)
    provider = PyVLLMProvider("/models/Qwen3.5-9B", tensor_parallel_size=2)
    result = await provider.generate("Solve: 2+2", model="Qwen3.5-9B")
    provider.close()
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

import httpx
from openai import AsyncOpenAI

from inference.exceptions import UnsupportedOperationError
from inference.provider import GenerationResult, InferenceProvider

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1")

logger = logging.getLogger(__name__)


# ── HTTP server provider ──────────────────────────────────────────────────────


class VLLMProvider(InferenceProvider):
    """InferenceProvider backed by a vLLM server with LoRA hot-loading support.

    Communicates with vLLM via two channels:
      - AsyncOpenAI SDK for generation (OpenAI-compatible endpoint).
      - httpx for LoRA adapter management (vLLM proprietary endpoints).

    Adapter tracking is maintained in an internal set to work around vLLM
    bug #11761 (list_lora_adapters unreliable after concurrent loads).

    Args:
        base_url: Override URL for the vLLM server.  Defaults to the
            ``VLLM_BASE_URL`` env var or ``http://localhost:8100/v1``.

    Example::

        provider = VLLMProvider(base_url="http://localhost:8100/v1")
        result = await provider.generate("def hello", model="Qwen2.5-Coder-7B")
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or VLLM_BASE_URL
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="not-needed-for-local-vllm",
        )
        self._loaded_adapters: set[str] = set()

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt, optionally using a loaded LoRA adapter.

        When ``adapter_id`` is provided it is passed as the model parameter to
        the OpenAI API — this is how vLLM routes to a loaded LoRA adapter.

        Args:
            prompt: The user-facing input prompt.
            model: Base model identifier.  Used as-is when no adapter is given.
            adapter_id: Name of a loaded LoRA adapter to apply.  When set,
                this value replaces ``model`` in the API call.
            max_tokens: Maximum number of tokens to generate.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature override.
            top_p: Nucleus sampling threshold override.
            repetition_penalty: Repetition penalty override.

        Returns:
            ``GenerationResult`` with the generated text and metadata.
        """
        effective_model = adapter_id if adapter_id is not None else model
        logger.debug(
            "generate: model=%s adapter_id=%s max_tokens=%d",
            effective_model,
            adapter_id,
            max_tokens,
        )

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = await self._client.chat.completions.create(
            model=effective_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            model=response.model,
            adapter_id=adapter_id,
            token_count=response.usage.total_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Load a LoRA adapter into the vLLM server.

        Posts to vLLM's ``/v1/load_lora_adapter`` endpoint.

        Args:
            adapter_id: Unique name for the adapter (used as ``lora_name``).
            adapter_path: Filesystem path to the adapter weights directory.

        Raises:
            httpx.HTTPStatusError: If the vLLM server returns an error.
        """
        url = f"{self._base_url.rstrip('/')}/load_lora_adapter"
        logger.debug("load_adapter: POST %s lora_name=%s", url, adapter_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"lora_name": adapter_id, "lora_path": adapter_path},
            )
            response.raise_for_status()

        self._loaded_adapters.add(adapter_id)
        logger.info("Adapter loaded: %s", adapter_id)

    async def unload_adapter(self, adapter_id: str) -> None:
        """Unload a LoRA adapter from the vLLM server.

        Posts to vLLM's ``/v1/unload_lora_adapter`` endpoint.

        Args:
            adapter_id: Name of the adapter to unload.

        Raises:
            httpx.HTTPStatusError: If the vLLM server returns an error.
        """
        url = f"{self._base_url.rstrip('/')}/unload_lora_adapter"
        logger.debug("unload_adapter: POST %s lora_name=%s", url, adapter_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"lora_name": adapter_id},
            )
            response.raise_for_status()

        self._loaded_adapters.discard(adapter_id)
        logger.info("Adapter unloaded: %s", adapter_id)

    async def list_adapters(self) -> list[str]:
        """List currently loaded LoRA adapters.

        Returns the internal tracking set rather than querying vLLM to avoid
        the unreliable list endpoint (vLLM bug #11761).

        Returns:
            Sorted list of adapter IDs currently tracked as loaded.
        """
        return sorted(self._loaded_adapters)


# ── In-process provider ───────────────────────────────────────────────────────


class PyVLLMProvider(InferenceProvider):
    """InferenceProvider backed by an in-process vLLM engine.

    Loads model + tokenizer directly via ``vllm.LLM`` — no server process
    needed.  Owns GPU memory; call ``close()`` to release it.

    Exposes three additional public helpers consumed by
    ``inference.benchmark_backends.VLLMBackend``:

    - ``stop_token_ids()`` — stop tokens for ``SamplingParams``
    - ``format_prompt(messages, enable_thinking, tools)`` — chat template
    - ``generate_batch_raw(prompts, sampling_params)`` — raw vLLM outputs

    These are intentionally public so benchmark wrappers can build
    batching and tool-calling on top without re-implementing tokenisation.

    LoRA hot-loading is not supported; those calls raise
    ``UnsupportedOperationError``.  Use ``VLLMProvider`` (HTTP) instead.

    Args:
        model_id: HuggingFace model ID or absolute local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Maximum KV-cache sequence length.
        dtype: Weight dtype (``"float16"``, ``"bfloat16"``, ``"auto"``).
        enable_prefix_caching: Enable prefix KV-cache sharing.
        language_model_only: Skip vision encoder init for hybrid checkpoints
            that ship without vision weights.

    Example::

        provider = PyVLLMProvider("/models/Qwen3.5-9B", tensor_parallel_size=2)
        result = await provider.generate("What is 2+2?", model="Qwen3.5-9B")
        print(result.text)
        provider.close()
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        dtype: str = "float16",
        enable_prefix_caching: bool = True,
        language_model_only: bool = True,
    ) -> None:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        from vllm import LLM  # type: ignore[import-untyped]

        logger.info(
            "Loading in-process vLLM model: %s  (tp=%d  dtype=%s)",
            model_id,
            tensor_parallel_size,
            dtype,
        )
        self._model_id = model_id
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._llm = LLM(
            model=model_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            trust_remote_code=True,
            language_model_only=language_model_only,
        )
        logger.info("In-process vLLM engine ready")

    # ── Helpers for benchmark wrappers ────────────────────────────────────────

    def stop_token_ids(self) -> list[int]:
        """Return stop token IDs for ``SamplingParams``.

        Includes ``<|im_end|>`` (Qwen/ChatML) and the tokenizer's EOS token.

        Returns:
            List of integer token IDs; ``None`` values filtered out.
        """
        im_end = self._tok.convert_tokens_to_ids("<|im_end|>")
        return [x for x in [im_end, self._tok.eos_token_id] if x is not None]

    def format_prompt(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Apply the tokeniser's chat template to a message list.

        Args:
            messages: OpenAI-style message dicts (role + content).
            enable_thinking: Enable Qwen3-style ``<think>…</think>`` prefix.
            tools: Tool-schema list forwarded to ``apply_chat_template`` for
                structured tool-call output.

        Returns:
            Formatted prompt string ready for vLLM generation.
        """
        return self._tok.apply_chat_template(  # type: ignore[no-any-return]
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )

    def generate_batch_raw(
        self,
        prompts: list[str],
        sampling_params: Any,
    ) -> list[Any]:
        """Run vLLM generation and return raw ``RequestOutput`` objects.

        Args:
            prompts: Pre-formatted strings (from ``format_prompt``).
            sampling_params: A ``vllm.SamplingParams`` instance.

        Returns:
            List of ``vllm.RequestOutput`` objects, one per prompt.
        """
        return self._llm.generate(prompts, sampling_params, use_tqdm=False)  # type: ignore[no-any-return]

    # ── InferenceProvider interface ───────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> GenerationResult:
        """Generate a single completion using the in-process vLLM engine.

        Drop-in for ``VLLMProvider.generate()`` in pipeline code that does
        not need LoRA hot-loading.

        Args:
            prompt: User-facing input text.
            model: Informational identifier echoed in the result.
            adapter_id: Ignored (interface compatibility).
            max_tokens: Maximum output tokens.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.  Defaults to near-zero (greedy).
            top_p: Nucleus sampling threshold.  Defaults to 1.0.
            repetition_penalty: Repetition penalty.  Defaults to 1.0.

        Returns:
            ``GenerationResult`` with the completion text and metadata.
        """
        from vllm import SamplingParams  # type: ignore[import-untyped]

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 1e-6,
            top_p=top_p if top_p is not None else 1.0,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else 1.0,
            stop_token_ids=self.stop_token_ids(),
        )

        outs = self._llm.generate([self.format_prompt(messages)], sp, use_tqdm=False)
        c = outs[0].outputs[0]
        return GenerationResult(
            text=c.text,
            model=model or self._model_id,
            adapter_id=adapter_id,
            token_count=len(c.token_ids),
            finish_reason=c.finish_reason or "stop",
        )

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Not supported — use ``VLLMProvider`` (HTTP) for LoRA workflows.

        Raises:
            UnsupportedOperationError: Always.
        """
        raise UnsupportedOperationError(
            "PyVLLMProvider does not support hot LoRA loading. "
            "Use VLLMProvider (HTTP server) for LoRA adapter workflows."
        )

    async def unload_adapter(self, adapter_id: str) -> None:
        """Not supported — use ``VLLMProvider`` (HTTP) for LoRA workflows.

        Raises:
            UnsupportedOperationError: Always.
        """
        raise UnsupportedOperationError(
            "PyVLLMProvider does not support hot LoRA unloading."
        )

    async def list_adapters(self) -> list[str]:
        """Return an empty list (no adapters managed in-process)."""
        return []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the vLLM engine and free GPU memory."""
        import torch  # type: ignore[import-untyped]

        logger.debug("Releasing in-process vLLM engine and CUDA cache")
        del self._llm
        gc.collect()
        torch.cuda.empty_cache()
