"""ModelWrapper: bridges step_node's model interface to the model layer stubs."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from rune.model.adapter import AdapterResult
from rune.model.adapter import hotswap_adapter as hotswap_adapter_fn
from rune.model.hypernetwork import generate_adapter_weights
from rune.model.inference import GenerationResult
from rune.model.inference import generate as inference_generate

if TYPE_CHECKING:
    from rune.config import PipelineConfig


class ModelWrapper:
    """Bridges step_node's model interface to the underlying model layer.

    Accepts a base_model, tokenizer, and hypernet already loaded, so the
    caller (or from_config) controls all GPU I/O.
    """

    def __init__(
        self,
        base_model: Any,
        tokenizer: Any,
        hypernet: Any,
        *,
        config: PipelineConfig,
    ) -> None:
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._hypernet = hypernet
        self._config = config
        self._layer_indices: list[int] = getattr(
            getattr(hypernet, "config", None), "layer_indices", []
        )

    @classmethod
    def from_config(cls, config: PipelineConfig) -> ModelWrapper:
        """Load model, tokenizer, and hypernet from config.

        All heavy imports (torch, transformers, peft) are deferred inside this
        method so the module stays importable in CPU-only CI.

        Args:
            config: Pipeline config; checkpoint_path must be non-empty.

        Returns:
            Initialised ModelWrapper.

        Raises:
            ValueError: If checkpoint_path is empty.
        """
        if not config.checkpoint_path:
            raise ValueError(
                "checkpoint_path must be set in config before calling from_config"
            )

        import torch  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        from rune.model.hypernetwork import (  # noqa: PLC0415
            HypernetworkConfig,
            load_hypernetwork,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        hypernet = load_hypernetwork(
            HypernetworkConfig(
                checkpoint_path=config.checkpoint_path,
                model_config_name=config.model_id,
            ),
            device=device,
        )

        hc = hypernet.config
        target_modules = list(hc.lora_config.target_modules)
        rank = hc.lora_config.r
        alpha = getattr(hc.lora_config, "lora_alpha", rank * 2)
        _raw_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=torch.bfloat16,
        ).to(device)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha * rank,
            target_modules=target_modules,
            lora_dropout=0.0,
        )
        base_model: Any = get_peft_model(_raw_model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        return cls(base_model, tokenizer, hypernet, config=config)

    def generate_adapter(
        self, trajectory_text: str, *, offload_base: bool = False,
    ) -> AdapterResult:
        """Generate LoRA weights from a trajectory via the hypernetwork.

        Args:
            trajectory_text: Serialised coding trajectory used as conditioning.
            offload_base: Move base model to CPU during the hypernetwork forward
                pass to free GPU memory.

        Returns:
            AdapterResult with a fresh UUID adapter_id and the generated state dict.
        """
        state_dict = generate_adapter_weights(
            hypernet=self._hypernet,
            trajectory_text=trajectory_text,
            base_model=self._base_model,
            tokenizer=self._tokenizer,
            layer_indices=self._layer_indices,
            offload_base=offload_base,
        )
        return AdapterResult(adapter_id=uuid.uuid4().hex, state_dict=state_dict)

    def hotswap_adapter(self, state_dict: dict[str, Any]) -> None:
        """Hot-swap LoRA weights into the base model in-place.

        Args:
            state_dict: PEFT-compatible adapter weights.
        """
        hotswap_adapter_fn(self._base_model, state_dict)

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        output_schema: type[Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        repetition_penalty: float = 1.1,
        top_p: float = 0.9,
        no_repeat_ngram_size: int = 0,
        thinking_budget: int = 1024,
    ) -> GenerationResult:
        return await inference_generate(
            self._base_model,
            self._tokenizer,
            prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            thinking_budget=thinking_budget,
        )
