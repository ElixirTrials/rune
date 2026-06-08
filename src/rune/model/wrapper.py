"""ModelWrapper: bridges step_node's model interface to the model layer stubs."""

from __future__ import annotations

import importlib.util
import logging
import uuid
from typing import TYPE_CHECKING, Any

from rune.model.adapter import AdapterResult, scale_lora_b
from rune.model.adapter import hotswap_adapter as hotswap_adapter_fn
from rune.model.hypernetwork import generate_adapter_weights
from rune.model.inference import GenerationResult
from rune.model.inference import generate as inference_generate
from rune.model.inference import (
    generate_continuation as inference_generate_continuation,
)

if TYPE_CHECKING:
    from rune.config import PipelineConfig


logger = logging.getLogger(__name__)


def resolve_attn_implementation(name: str) -> str:
    """Fall back to ``sdpa`` when ``flash_attention_2`` is configured but absent.

    flash-attn is an optional, environment-specific wheel; the model profile
    pins it for speed, but it must not hard-fail a run on a box where it is not
    installed (sdpa is a numerically-equivalent drop-in for our purposes).
    """
    if name == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        logger.warning(
            "flash_attention_2 requested but flash_attn is not installed; "
            "falling back to sdpa"
        )
        return "sdpa"
    return name


def peft_scaling_params(
    checkpoint_alpha: float, rank: int, use_bias: bool
) -> tuple[int, float]:
    """Compute the engine's PEFT ``(r, lora_alpha)`` from the checkpoint contract.

    PEFT applies the LoRA delta as ``delta * (lora_alpha / r)``. To realize the
    shared contract's effective scaling (the RAW checkpoint ``lora_alpha``,
    applied un-divided; see ``rune.model.adapter_contract.effective_scaling``),
    we pick ``lora_alpha_peft = checkpoint_alpha * r_peft`` so the quotient is
    exactly ``checkpoint_alpha`` and uniform across all ranks.

    When the hypernet was trained ``use_bias``, ``generate_adapter_weights``
    emits a ``combine_lora``-assembled state_dict whose rank axis is doubled to
    ``2r`` (ranks ``0..r-1`` context A/B, ``r..2r-1`` head bias), so the PEFT
    adapter must be built at ``2r`` or hot-swap misapplies (the rank-16 crash).

    This is the ONE place the engine's PEFT sizing is derived; both
    ``from_config`` and its unit test call it so the formula can't drift.

    Returns:
        ``(r_peft, lora_alpha_peft)``.
    """
    r_peft = 2 * rank if use_bias else rank
    return r_peft, checkpoint_alpha * r_peft


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
        # Sakana's contract (rune.model.adapter_contract.effective_scaling):
        # the effective LoRA scaling is the RAW checkpoint lora_alpha, applied
        # un-divided in ctx_to_lora.lora_forward (NOT alpha/r). The prior code
        # set PEFT (r=rank, lora_alpha=alpha) -> PEFT scaling alpha/r, which is
        # 8x too weak at r=8 and collapsed recall.
        checkpoint_alpha = float(getattr(hc.lora_config, "lora_alpha", rank * 2))
        use_bias = bool(getattr(hc, "use_bias", False))
        r_peft, lora_alpha_peft = peft_scaling_params(checkpoint_alpha, rank, use_bias)
        # dtype + attention impl come from the model profile (config), not
        # hardcoded — different models need different generation contracts.
        _raw_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=getattr(torch, config.dtype),
            attn_implementation=resolve_attn_implementation(config.attn_implementation),
        ).to(device)
        lora_config = LoraConfig(
            r=r_peft,
            lora_alpha=lora_alpha_peft,
            target_modules=target_modules,
            lora_dropout=0.0,
            use_rslora=False,
        )
        base_model: Any = get_peft_model(_raw_model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        return cls(base_model, tokenizer, hypernet, config=config)

    def generate_adapter(
        self,
        trajectory_text: str,
        *,
        offload_base: bool = False,
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

    def count_tokens(self, text: str) -> int:
        """Token count of ``text`` under the base tokenizer (content tokens only).

        The engine step logs adapter-conditioning tokens vs prompt tokens per
        turn (issue #52 GOAL-3 pre-reg g): the adapter-as-memory thesis
        instrument — the prompt stays ~flat while the adapter's trajectory
        conditioning grows. ``add_special_tokens=False`` so the count reflects
        content, comparable across the two surfaces.
        """
        return len(self._tokenizer(text, add_special_tokens=False).input_ids)

    def hotswap_adapter(self, state_dict: dict[str, Any]) -> None:
        """Hot-swap LoRA weights into the base model in-place.

        Args:
            state_dict: PEFT-compatible adapter weights.
        """
        hotswap_adapter_fn(self._base_model, state_dict)

    def reset_adapter(self) -> None:
        """Zero LoRA weights so a prior task's adapter cannot bleed into the next."""
        from peft import get_peft_model_state_dict  # noqa: PLC0415

        sd: dict[str, Any] = get_peft_model_state_dict(self._base_model)  # type: ignore[no-untyped-call]
        hotswap_adapter_fn(self._base_model, scale_lora_b(sd, 0.0))

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
        presence_penalty: float = 0.0,
        thinking_budget: int = 0,  # 0 = non-thinking; config drives the real value
        skip_completion_retry: bool = False,
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
            presence_penalty=presence_penalty,
            thinking_budget=thinking_budget,
            skip_completion_retry=skip_completion_retry,
        )

    async def generate_continuation(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prefix: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        repetition_penalty: float = 1.1,
        top_p: float = 0.9,
        no_repeat_ngram_size: int = 0,
        presence_penalty: float = 0.0,
    ) -> GenerationResult:
        return await inference_generate_continuation(
            self._base_model,
            self._tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_prefix=assistant_prefix,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            presence_penalty=presence_penalty,
        )
