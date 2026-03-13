"""TransformersProvider: InferenceProvider using HuggingFace transformers + PEFT.

Loads models via AutoModelForCausalLM and applies LoRA adapters via PEFT.
This is the only provider that natively supports PEFT-format adapters
(safetensors) as output by the hypernetwork.

IMPORTANT: transformers, torch, and peft are imported inside method bodies
per INFRA-05 pattern so that this module is importable in CPU-only CI.
"""

from __future__ import annotations

import logging
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

    def _load_model_if_needed(self) -> None:
        """Load the base model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        logger.info("Loading model: %s (device=%s)", self._model_name, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=self._torch_dtype,
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
    ) -> GenerationResult:
        """Generate text using transformers with optional PEFT adapter.

        Args:
            prompt: The input prompt.
            model: Ignored (model is set at construction).
            adapter_id: LoRA adapter ID to activate. Must be loaded via
                load_adapter() before use.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with generated text and metadata.

        Raises:
            ValueError: If adapter_id is provided but has not been loaded.
        """
        import torch  # noqa: PLC0415

        self._load_model_if_needed()

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

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_tokens = outputs.shape[1]

        return GenerationResult(
            text=text,
            model=self._model_name,
            adapter_id=self._active_adapter,
            token_count=total_tokens,
            finish_reason="stop",
        )

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

        if self._active_adapter:
            # Already has a PEFT wrapper — load additional adapter
            self._model.load_adapter(adapter_path, adapter_name=adapter_id)
            self._model.set_adapter(adapter_id)
        else:
            # First adapter — wrap base model with PeftModel
            self._model = PeftModel.from_pretrained(
                self._base_model, adapter_path, adapter_name=adapter_id
            )
            self._model.to(self._device)
            self._model.eval()

        self._active_adapter = adapter_id
        logger.info("Activated adapter: %s", adapter_id)

    def _deactivate_adapter(self) -> None:
        """Deactivate current adapter, revert to base model."""
        if self._active_adapter and self._base_model is not None:
            self._model = self._base_model
            self._active_adapter = None
            logger.info("Deactivated adapter, reverted to base model")

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
        """Remove a registered adapter.

        Args:
            adapter_id: The adapter name to remove.
        """
        if adapter_id in self._loaded_adapters:
            if self._active_adapter == adapter_id:
                self._deactivate_adapter()
            del self._loaded_adapters[adapter_id]
            logger.info("Unloaded adapter %s", adapter_id)

    async def list_adapters(self) -> list[str]:
        """List all registered adapter IDs.

        Returns:
            Sorted list of registered adapter IDs.
        """
        return sorted(self._loaded_adapters.keys())
