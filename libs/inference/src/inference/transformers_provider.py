"""TransformersProvider: InferenceProvider using HuggingFace transformers + PEFT.

Loads models via AutoModelForCausalLM and applies LoRA adapters via PEFT.
This is the only provider that natively supports PEFT-format adapters
(safetensors) as output by the hypernetwork.

IMPORTANT: transformers, torch, and peft are imported inside method bodies
per INFRA-05 pattern so that this module is importable in CPU-only CI.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from inference.provider import GenerationResult, InferenceProvider

logger = logging.getLogger(__name__)


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
        >>> provider = TransformersProvider(model_name="Qwen/Qwen2.5-Coder-0.5B")
        >>> result = await provider.generate("def hello", model="ignored")
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "cpu",
        torch_dtype: str = "auto",
    ) -> None:
        """Initialize TransformersProvider.

        Args:
            model_name: HuggingFace model ID or local path.
            device: Device to load model onto.
            torch_dtype: Model dtype string.
        """
        self._model_name = model_name
        self._device = device
        self._torch_dtype = torch_dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._base_model: Any = None
        self._loaded_adapters: dict[str, str] = {}  # id -> path
        self._active_adapter: str | None = None
        self._is_peft_wrapped: bool = False

    def _load_model_if_needed(self) -> None:
        """Load the base model and tokenizer if not already loaded."""
        if self._model is not None:
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
            param_count = getattr(config, "num_parameters", None)
            if param_count is None:
                # Estimate from config fields
                h = getattr(config, "hidden_size", 2048)
                v = getattr(config, "vocab_size", 32000)
                n = getattr(config, "num_hidden_layers", 24)
                param_count = v * h + n * 12 * h * h
            resolved_dtype = resolve_model_dtype(  # type: ignore[assignment]
                param_count=param_count, device=self._device
            )
            logger.info("Inference model dtype resolved to %s", resolved_dtype)

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=resolved_dtype,
        )
        self._model.to(self._device)
        self._model.eval()
        self._base_model = self._model
        logger.info("Model loaded: %s", self._model_name)

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
        formatted = self._format_prompt(prompt, system_prompt)
        inputs = self._tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=8192
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_tokens = outputs.shape[1]
        new_token_count = len(new_tokens)

        # Detect truncation: generated exactly max_tokens means cut off
        finish_reason = "length" if new_token_count >= max_tokens else "stop"

        return GenerationResult(
            text=text,
            model=self._model_name,
            adapter_id=self._active_adapter,
            token_count=total_tokens,
            finish_reason=finish_reason,
        )

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        """Format prompt using the tokenizer's chat template when available.

        Constructs a messages list and applies the tokenizer's chat template
        so instruction-tuned models receive properly structured input.
        Falls back to plain concatenation when no chat template exists.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "user", "content": f"{system_prompt}\n\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Tokenizer has no chat template — fall back to raw concat
            if system_prompt:
                return f"{system_prompt}\n\n{prompt}"
            return prompt

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
        adapter_config.json as output by save_hypernetwork_adapter().

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
            # If no adapters remain, revert to base model
            if not self._loaded_adapters and self._is_peft_wrapped:
                self._model = self._base_model
                self._is_peft_wrapped = False
                logger.info("All adapters removed, reverted to base model")
            else:
                logger.info("Unloaded adapter %s", adapter_id)

    async def list_adapters(self) -> list[str]:
        """List all registered adapter IDs.

        Returns:
            Sorted list of registered adapter IDs.
        """
        return sorted(self._loaded_adapters.keys())
