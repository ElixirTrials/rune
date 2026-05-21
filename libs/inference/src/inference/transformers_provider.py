"""TransformersProvider: InferenceProvider using HuggingFace transformers + PEFT.

Loads models via AutoModelForCausalLM and applies LoRA adapters via PEFT.
This is the only provider that natively supports PEFT-format adapters
(safetensors) as output by the hypernetwork.

IMPORTANT: transformers, torch, and peft are imported inside method bodies
per INFRA-05 pattern so that this module is importable in CPU-only CI.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, cast

from pydantic import BaseModel

from inference.provider import GenerationResult, InferenceProvider

logger = logging.getLogger(__name__)


def _config_int(cfg: object, attr: str, default: int) -> int:
    """Read a positive integer from a HuggingFace config object."""
    raw = getattr(cfg, attr, None)
    return default if raw is None else int(raw)


class _ThinkingBudgetProcessor:
    """Forces </think> emission once the thinking token budget is exhausted."""

    def __init__(self, end_think_token_id: int, budget: int, prompt_len: int) -> None:
        self._etid = end_think_token_id
        self._budget = budget
        self._prompt_len = prompt_len
        self._done = False

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self._done:
            return scores

        new_ids = input_ids[0, self._prompt_len :]

        if (new_ids == self._etid).any():
            self._done = True
            return scores

        if new_ids.shape[0] >= self._budget:
            scores.fill_(float("-inf"))
            scores[:, self._etid] = 0.0

        return scores


class TransformersProvider(InferenceProvider):
    """InferenceProvider backed by HuggingFace transformers with PEFT LoRA.

    Loads models locally via AutoModelForCausalLM. Adapters are applied
    via PEFT's PeftModel, which natively reads the safetensors format
    output by the hypernetwork.

    Args:
        model_name: HuggingFace model ID or local path.
        device: Device to load model onto ('cpu', 'mps', 'cuda').
        torch_dtype: Model dtype ('auto', 'float16', 'bfloat16').

    Example:
        >>> provider = TransformersProvider(model_name="Qwen/Qwen3.5-9B")
        >>> result = await provider.generate("def hello", model="ignored")
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        torch_dtype: str = "auto",
        pool: Any = None,
    ) -> None:
        """Initialize TransformersProvider.

        Args:
            model_name: HuggingFace model ID or local path.
            device: Device to load model onto.
            torch_dtype: Model dtype string.
            pool: Optional ModelPool instance. When provided, the base model
                and tokenizer are borrowed from the pool instead of loaded
                independently, avoiding a second copy in VRAM.
        """
        self._model_name = model_name
        self._device = device
        self._torch_dtype = torch_dtype
        self._pool = pool
        self._model: Any = None
        self._tokenizer: Any = None
        self._base_model: Any = None
        self._loaded_adapters: dict[str, str] = {}  # id -> path
        self._active_adapter: str | None = None
        self._is_peft_wrapped: bool = False
        self._think_token_id: int | None = None
        self._end_think_token_id: int | None = None
        self._think_ids_resolved: bool = False
        self._xgr_compiler: Any = None
        self._xgr_compiled_cache: dict[type[BaseModel], Any] = {}

    def _load_model_if_needed(self) -> None:
        """Load the base model and tokenizer if not already loaded.

        When a pool is available, borrows the model and tokenizer from it
        instead of loading a second copy into VRAM.
        """
        if self._model is not None:
            return

        if self._pool is not None:
            model, tokenizer = self._pool.base_model()
            # Clean residual PEFT state left by a previous pipeline run
            if hasattr(model, "peft_config"):
                from peft import PeftModel as _PeftModel  # noqa: PLC0415

                if isinstance(model, _PeftModel):
                    # PeftModel stubs type base_model as
                    # Tensor | Module; unload() is safe.
                    model = cast(Any, model).unload()
                    self._pool._model = model
                elif hasattr(model, "peft_config"):
                    del model.peft_config
                logger.info("Cleaned residual PEFT state from pooled model")
            self._model = model
            self._tokenizer = tokenizer
            self._base_model = model
            self._device = self._pool.device
            logger.info("Borrowed base model from pool (device=%s)", self._device)
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        logger.info("Loading model: %s (device=%s)", self._model_name, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Resolve dtype: prefer fp32 on GPU when VRAM allows (better generation
        # quality for the resident inference model), fall back to model default.
        resolved_dtype = self._torch_dtype
        if resolved_dtype == "auto" and self._device != "cpu":
            from shared.hardware import resolve_model_dtype  # noqa: PLC0415
            from transformers import AutoConfig  # noqa: PLC0415

            config = AutoConfig.from_pretrained(self._model_name)
            raw_param_count = getattr(config, "num_parameters", None)
            if raw_param_count is None:
                # Multimodal configs (e.g. Qwen3.5) nest dims under text_config
                text_cfg = getattr(config, "text_config", config)
                h = _config_int(
                    text_cfg, "hidden_size", _config_int(config, "hidden_size", 2048)
                )
                v = _config_int(
                    text_cfg, "vocab_size", _config_int(config, "vocab_size", 32000)
                )
                n = _config_int(
                    text_cfg,
                    "num_hidden_layers",
                    _config_int(config, "num_hidden_layers", 24),
                )
                param_count = v * h + n * 12 * h * h
            else:
                param_count = int(raw_param_count)
            resolved_dtype = resolve_model_dtype(  # type: ignore[assignment]
                param_count=param_count, device=self._device
            )
            logger.info("Inference model dtype resolved to %s", resolved_dtype)

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            dtype=resolved_dtype,
        )
        self._model.to(self._device)
        self._model.eval()
        self._base_model = self._model
        logger.info("Model loaded: %s", self._model_name)

    def _init_xgr_compiler(self) -> None:
        """Create and cache the XGrammar compiler from the loaded tokenizer."""
        import xgrammar as xgr  # noqa: PLC0415

        config = self._model.config
        vocab_size = getattr(
            getattr(config, "text_config", config),
            "vocab_size",
            self._tokenizer.vocab_size,
        )
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self._tokenizer, vocab_size=vocab_size
        )
        self._xgr_compiler = xgr.GrammarCompiler(tokenizer_info)
        logger.info("XGrammar compiler initialized (vocab_size=%d)", vocab_size)

    def _get_xgr_compiler(self) -> Any:
        """Lazily initialize and return the XGrammar compiler."""
        if self._xgr_compiler is None:
            self._init_xgr_compiler()
        return self._xgr_compiler

    async def generate(  # noqa: C901
        self,
        prompt: str,
        model: str,
        adapter_id: str | None = None,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool = True,
        thinking_budget: int = 0,
        json_schema: type[BaseModel] | None = None,
    ) -> GenerationResult:
        """Generate text using transformers with optional PEFT adapter.

        Args:
            prompt: The user-facing input prompt.
            model: Ignored (model is set at construction).
            adapter_id: LoRA adapter ID to activate. Must be loaded via
                load_adapter() before use.
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system-level instruction prepended via
                the tokenizer's chat template when available.
            temperature: Sampling temperature (default from pipeline config).
            top_p: Nucleus sampling threshold (default from pipeline config).
            repetition_penalty: Repetition penalty (default 1.0 = off).
            enable_thinking: Whether to allow model thinking/reasoning
                tokens. When False, passes ``enable_thinking=False`` to the
                chat template and suppresses think token generation via
                ``suppress_tokens`` so output goes entirely to the response.
            thinking_budget: Extra token headroom for thinking when enabled.
            json_schema: Optional Pydantic model class for constrained JSON output.

        Returns:
            GenerationResult with generated text and metadata.

        Raises:
            ValueError: If adapter_id is provided but has not been loaded.
        """
        import torch  # noqa: PLC0415

        self._load_model_if_needed()

        # Apply defaults from pipeline config
        if temperature is None:
            temperature = float(os.environ.get("RUNE_TEMPERATURE", "0.25"))
        if top_p is None:
            top_p = float(os.environ.get("RUNE_TOP_P", "0.9"))
        if repetition_penalty is None:
            repetition_penalty = float(
                os.environ.get("RUNE_REPETITION_PENALTY", "1.04")
            )

        # Validate adapter before switching
        if adapter_id and adapter_id not in self._loaded_adapters:
            raise ValueError(
                f"Adapter '{adapter_id}' has not been loaded. "
                "Call load_adapter() first."
            )

        # Switch adapter if needed
        if adapter_id and adapter_id != self._active_adapter:
            self._activate_adapter(adapter_id)
        elif not adapter_id and self._active_adapter:
            self._deactivate_adapter()

        # Build chat-formatted prompt via tokenizer's chat template
        formatted = self._format_prompt(prompt, system_prompt, enable_thinking)
        inputs = self._tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=8192
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        effective_max = max_tokens + thinking_budget
        gen_kwargs: dict[str, object] = {
            "max_new_tokens": effective_max,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        self._resolve_think_token_ids()
        if not enable_thinking and self._think_token_id is not None:
            suppress = [self._think_token_id]
            if self._end_think_token_id is not None:
                suppress.append(self._end_think_token_id)
            gen_kwargs["suppress_tokens"] = suppress

        if (
            enable_thinking
            and thinking_budget > 0
            and self._end_think_token_id is not None
        ):
            gen_kwargs["logits_processor"] = [
                _ThinkingBudgetProcessor(
                    end_think_token_id=self._end_think_token_id,
                    budget=thinking_budget,
                    prompt_len=input_len,
                )
            ]

        if json_schema is not None:
            if enable_thinking and thinking_budget > 0:
                raise ValueError(
                    "json_schema and thinking_budget>0 are mutually exclusive: "
                    "XGrammar constraints would corrupt thinking tokens"
                )
            import xgrammar as xgr  # noqa: PLC0415

            compiled = self._xgr_compiled_cache.get(json_schema)
            if compiled is None:
                compiled = self._get_xgr_compiler().compile_json_schema(
                    json.dumps(json_schema.model_json_schema())
                )
                self._xgr_compiled_cache[json_schema] = compiled
            xgr_processor = xgr.contrib.hf.LogitsProcessor(compiled)
            if "logits_processor" in gen_kwargs:
                gen_kwargs["logits_processor"].append(xgr_processor)  # type: ignore[attr-defined]
            else:
                gen_kwargs["logits_processor"] = [xgr_processor]

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_token_ids = outputs[0][input_len:].tolist()
        total_tokens = outputs.shape[1]
        new_token_count = len(new_token_ids)
        finish_reason = "length" if new_token_count >= effective_max else "stop"

        text, thinking = self._split_thinking(new_token_ids, enable_thinking)

        return GenerationResult(
            text=text,
            model=self._model_name,
            adapter_id=self._active_adapter,
            token_count=total_tokens,
            finish_reason=finish_reason,
            thinking=thinking,
        )

    def _format_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
        enable_thinking: bool = True,
    ) -> str:
        """Format prompt using the tokenizer's chat template when available.

        Constructs a messages list and applies the tokenizer's chat template
        so instruction-tuned models receive properly structured input.
        Falls back to plain concatenation when no chat template exists.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Template doesn't support enable_thinking kwarg — retry without it
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (AttributeError, TypeError, ValueError):
                logger.warning(
                    "Chat template failed, falling back to raw concat",
                    exc_info=True,
                )
                if system_prompt:
                    return f"{system_prompt}\n\n{prompt}"
                return prompt
        except (AttributeError, ValueError):
            logger.warning(
                "Chat template failed, falling back to raw concat",
                exc_info=True,
            )
            if system_prompt:
                return f"{system_prompt}\n\n{prompt}"
            return prompt

    def _resolve_think_token_ids(self) -> None:
        """Cache ``<think>`` and ``</think>`` token IDs from the tokenizer."""
        if self._think_ids_resolved:
            return
        if self._tokenizer is None:
            return
        tid = self._tokenizer.convert_tokens_to_ids("<think>")
        etid = self._tokenizer.convert_tokens_to_ids("</think>")
        if isinstance(tid, int) and tid != self._tokenizer.unk_token_id:
            self._think_token_id = tid
            self._end_think_token_id = etid if isinstance(etid, int) else None
        else:
            self._think_token_id = None
            self._end_think_token_id = None
        self._think_ids_resolved = True

    def _split_thinking(
        self,
        token_ids: list[int],
        enable_thinking: bool,
    ) -> tuple[str, str | None]:
        """Separate thinking from response using token IDs, not regex.

        When ``enable_thinking`` is True, finds the last ``</think>`` token
        and splits: everything before is thinking, everything after is the
        response.  If ``</think>`` is missing, all output is treated as
        thinking (returns empty text) to prevent deliberation from leaking
        into downstream phases.
        """
        etid = self._end_think_token_id
        if not enable_thinking or etid is None:
            text = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            return text, None

        try:
            idx = len(token_ids) - token_ids[::-1].index(etid)
        except ValueError:
            # No </think> found — all tokens are unclosed thinking.
            thinking_text = self._tokenizer.decode(
                token_ids, skip_special_tokens=True
            ).strip()
            if thinking_text:
                logger.warning(
                    "enable_thinking=True but no </think> found; "
                    "treating %d tokens as thinking (text will be empty)",
                    len(token_ids),
                )
            return "", thinking_text or None

        thinking_text = self._tokenizer.decode(
            token_ids[:idx], skip_special_tokens=True
        ).strip()
        content_text = self._tokenizer.decode(
            token_ids[idx:], skip_special_tokens=True
        ).strip()
        return content_text, thinking_text or None

    def _activate_adapter(self, adapter_id: str) -> None:
        """Activate a loaded PEFT adapter.

        Args:
            adapter_id: The adapter name to activate. Must already be in
                ``_loaded_adapters``.

        Raises:
            ValueError: If the adapter has not been loaded.
        """
        if adapter_id not in self._loaded_adapters:
            raise ValueError(
                f"Adapter '{adapter_id}' has not been loaded. "
                "Call load_adapter() first."
            )

        from peft import PeftModel  # noqa: PLC0415

        adapter_path = self._loaded_adapters[adapter_id]

        if self._is_peft_wrapped:
            # Already has a PEFT wrapper — check if adapter is already loaded
            if adapter_id not in self._model.peft_config:
                self._model.load_adapter(adapter_path, adapter_name=adapter_id)
            self._model.enable_adapter_layers()
            self._model.set_adapter(adapter_id)
        else:
            # First adapter — wrap base model with PeftModel
            self._model = PeftModel.from_pretrained(
                self._base_model, adapter_path, adapter_name=adapter_id
            )
            self._model.to(self._device)
            self._model.eval()
            self._is_peft_wrapped = True

        self._active_adapter = adapter_id
        logger.info("Activated adapter: %s", adapter_id)

    def _deactivate_adapter(self) -> None:
        """Deactivate current adapter, keeping PeftModel wrapper alive."""
        if self._active_adapter and self._is_peft_wrapped:
            self._model.disable_adapter_layers()
            self._active_adapter = None
            logger.info("Deactivated adapter layers (wrapper preserved)")

    async def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Register a PEFT adapter directory for use during generation.

        The adapter directory must contain adapter_model.safetensors and
        adapter_config.json in standard PEFT format.

        Args:
            adapter_id: Unique name for the adapter.
            adapter_path: Path to the PEFT adapter directory.
        """
        self._loaded_adapters[adapter_id] = adapter_path
        logger.info("Registered adapter %s -> %s", adapter_id, adapter_path)

    async def unload_adapter(self, adapter_id: str) -> None:
        """Remove a registered adapter, freeing GPU memory.

        Args:
            adapter_id: The adapter name to remove.
        """
        if adapter_id in self._loaded_adapters:
            if self._active_adapter == adapter_id:
                self._deactivate_adapter()
            # Delete from PeftModel to free GPU memory
            if self._is_peft_wrapped and adapter_id in self._model.peft_config:
                self._model.delete_adapter(adapter_id)
            del self._loaded_adapters[adapter_id]
            # If no adapters remain, fully unwrap PEFT layers from the base model
            if not self._loaded_adapters and self._is_peft_wrapped:
                clean_model = self._model.base_model.unload()
                self._model = clean_model
                self._base_model = clean_model
                self._is_peft_wrapped = False
                logger.info("All adapters removed, PEFT layers unwrapped")
            else:
                logger.info("Unloaded adapter %s", adapter_id)

    async def list_adapters(self) -> list[str]:
        """List all registered adapter IDs.

        Returns:
            Sorted list of registered adapter IDs.
        """
        return sorted(self._loaded_adapters.keys())
