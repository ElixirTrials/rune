"""Resident GPU model pool for base model and hypernetwork.

Keeps the base LLM (e.g. Qwen 9B) and hypernetwork (HyperLoRA perceiver)
resident in GPU memory across pipeline runs, eliminating repeated
load/unload cycles (~10 per problem without pooling).

Ownership: ModelPool is the single owner of these GPU tensors. All other
modules borrow references via base_model() and hypernetwork().

IMPORTANT: All GPU/ML imports (torch, transformers, etc.) are deferred
inside method bodies per INFRA-05 — this module is importable in CPU-only CI.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_POOL: ModelPool | None = None


class ModelPool:
    """Singleton owner of GPU-resident base model and hypernetwork.

    Use the ``create()`` factory to build an instance, then register it
    with ``set_pool()`` so the rest of the pipeline can retrieve it via
    ``get_pool()``.

    Attributes:
        model_name: HuggingFace model ID or local path of the base model.
        device: Compute device the models are loaded onto.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        hypernet_checkpoint_path: str | None,
        hypernet_variant: str,
    ) -> None:
        """Initialise a ModelPool (prefer ``create()`` over direct construction).

        Args:
            model_name: HuggingFace model ID or local path.
            device: Target device (``"cuda"``, ``"cpu"``, ``"mps"``).
            hypernet_checkpoint_path: Local path to hypernetwork checkpoint.
                ``None`` triggers HuggingFace download on first use.
            hypernet_variant: HuggingFace variant name used when downloading.
        """
        self._model_name = model_name
        self._device = device
        self._hypernet_checkpoint_path = hypernet_checkpoint_path
        self._hypernet_variant = hypernet_variant

        self._model: Any = None
        self._tokenizer: Any = None
        self._hypernet: Any = None
        self._hypernet_config: Any = None
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        model_name: str,
        device: str = "cuda",
        hypernet_checkpoint_path: str | None = None,
        hypernet_variant: str = "gemma_demo",
    ) -> "ModelPool":
        """Create a ModelPool without loading any weights.

        Args:
            model_name: HuggingFace model ID or local path for the base LLM.
            device: Device to load models onto (default ``"cuda"``).
            hypernet_checkpoint_path: Path to a local hypernetwork checkpoint.
                Downloads from HuggingFace when ``None``.
            hypernet_variant: Variant name used for HuggingFace download.

        Returns:
            A freshly constructed ModelPool ready for lazy loading.
        """
        return cls(
            model_name=model_name,
            device=device,
            hypernet_checkpoint_path=hypernet_checkpoint_path,
            hypernet_variant=hypernet_variant,
        )

    @property
    def model_name(self) -> str:
        """HuggingFace model ID or local path of the base model."""
        return self._model_name

    @property
    def device(self) -> str:
        """Compute device models are loaded onto."""
        return self._device

    def base_model(self) -> tuple[Any, Any]:
        """Return the resident base model and tokenizer, loading on first call.

        On the first call, loads the model via ``AutoModelForCausalLM`` and
        the tokenizer via ``AutoTokenizer``, resolves the optimal dtype for
        the target device, moves the model to ``device``, and puts it into
        eval mode.  When the model in bf16 won't fit in available VRAM
        (e.g. hypernetwork already loaded), falls back to NF4 4-bit
        quantization automatically.  Subsequent calls return the cached
        objects without re-loading.

        Returns:
            Tuple of ``(model, tokenizer)``.
        """
        if self._model is not None:
            return self._model, self._tokenizer

        with self._lock:
            if self._model is not None:
                return self._model, self._tokenizer

            from transformers import (  # noqa: PLC0415
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            logger.info(
                "ModelPool: loading base model %s on %s", self._model_name, self._device
            )

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            param_count = self._estimate_param_count()

            model: Any
            if self._should_quantize(param_count):
                model = self._load_quantized()
            else:
                dtype = self._pick_dtype(param_count)
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    dtype=dtype,
                )
                model.to(self._device)

            model.eval()
            self._model = model
            self._tokenizer = tokenizer
            logger.info("ModelPool: base model loaded")
        return self._model, self._tokenizer

    def _estimate_param_count(self) -> int:
        """Estimate model parameter count from its config."""
        from transformers import AutoConfig  # noqa: PLC0415

        config = AutoConfig.from_pretrained(self._model_name)
        param_count = getattr(config, "num_parameters", None)
        if callable(param_count):
            param_count = None
        if param_count is None:
            text_cfg = getattr(config, "text_config", config)

            def _cfg(name: str, default: int) -> int:
                val = getattr(text_cfg, name, None) or getattr(config, name, default)
                return default if val is None else int(val)

            h = _cfg("hidden_size", 2048)
            v = _cfg("vocab_size", 32000)
            n = _cfg("num_hidden_layers", 24)
            i = _cfg("intermediate_size", 4 * h)
            n_q = _cfg("num_attention_heads", h // 128)
            n_kv = _cfg("num_key_value_heads", n_q)
            head_dim = h // n_q if n_q else 128

            attn = (n_q + 2 * n_kv) * head_dim * h + n_q * head_dim * h
            ffn = 3 * h * i
            lm_head = 0 if getattr(config, "tie_word_embeddings", True) else v * h
            param_count = v * h + n * (attn + ffn) + lm_head

        logger.info(
            "ModelPool: estimated param_count=%s (%.1fB)",
            param_count,
            param_count / 1e9,
        )
        return param_count

    def _pick_dtype(self, param_count: int) -> Any:
        """Pick the highest-precision dtype that fits available VRAM."""
        from shared.hardware import resolve_model_dtype  # noqa: PLC0415

        dtype = resolve_model_dtype(param_count=param_count, device=self._device)
        logger.info("ModelPool: resolved dtype=%s", dtype)
        return dtype

    def _should_quantize(self, param_count: int) -> bool:
        """Return True when the bf16 model won't fit in available VRAM.

        Checks the ``RUNE_POOL_QUANTIZE`` env var first (``1``/``0``), then
        auto-detects by comparing estimated bf16 weight size against free
        VRAM with a small headroom margin.
        """
        import os  # noqa: PLC0415

        override = os.environ.get("RUNE_POOL_QUANTIZE", "").lower()
        if override in ("1", "true"):
            logger.info("ModelPool: quantization forced via RUNE_POOL_QUANTIZE")
            return True
        if override in ("0", "false"):
            return False

        if not self._device.startswith("cuda"):
            return False

        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False

        dev_idx = int(self._device.split(":")[1]) if ":" in self._device else 0
        total = torch.cuda.get_device_properties(dev_idx).total_memory
        allocated = torch.cuda.memory_allocated(dev_idx)
        free = total - allocated
        bf16_bytes = param_count * 2

        # 15% headroom for activations / CUDA allocator overhead
        fits = bf16_bytes < free * 0.85
        if not fits:
            logger.info(
                "ModelPool: bf16 model (%.1f GiB) exceeds available VRAM "
                "(%.1f GiB free of %.1f GiB) — falling back to NF4 4-bit",
                bf16_bytes / (1 << 30),
                free / (1 << 30),
                total / (1 << 30),
            )
        return not fits

    def _load_quantized(self) -> Any:
        """Load the base model with NF4 4-bit quantization via bitsandbytes."""
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            quantization_config=bnb_config,
            device_map={"": self._device},
        )
        logger.info("ModelPool: base model loaded with NF4 4-bit quantization")
        return model

    def hypernetwork(self) -> tuple[Any, Any]:
        """Return the resident hypernetwork and its config, loading on first call.

        Uses ``load_hypernetwork`` from ``model_training.hypernetwork``.
        Subsequent calls return the cached objects without re-loading.

        Returns:
            Tuple of ``(hypernet, hypernet_config)``.
        """
        if self._hypernet is not None:
            return self._hypernet, self._hypernet_config

        with self._lock:
            if self._hypernet is not None:
                return self._hypernet, self._hypernet_config

            from model_training.hypernetwork import load_hypernetwork  # noqa: PLC0415

            logger.info("ModelPool: loading hypernetwork on %s", self._device)
            hypernet, hc = load_hypernetwork(
                checkpoint_path=self._hypernet_checkpoint_path,
                variant=self._hypernet_variant,
                device=self._device,
            )
            self._hypernet = hypernet
            self._hypernet_config = hc
            logger.info("ModelPool: hypernetwork loaded")
        return self._hypernet, self._hypernet_config

    def release(self) -> None:
        """Clear all cached models so the next call reloads from scratch.

        Does not call torch.cuda.empty_cache() — callers that need explicit
        VRAM reclamation should do so themselves after release().
        """
        self._model = None
        self._tokenizer = None
        self._hypernet = None
        self._hypernet_config = None
        logger.info("ModelPool: cache cleared")


def get_pool() -> ModelPool:
    """Return the process-wide ModelPool singleton.

    Returns:
        The registered ModelPool.

    Raises:
        RuntimeError: If ``set_pool()`` has not been called yet.
    """
    if _POOL is None:
        raise RuntimeError(
            "ModelPool not initialised — call set_pool() before get_pool()."
        )
    return _POOL


def set_pool(pool: ModelPool) -> None:
    """Register a ModelPool as the process-wide singleton.

    Releases the previous pool (if any) before replacing it.

    Args:
        pool: The ModelPool instance to register.
    """
    global _POOL  # noqa: PLW0603
    if _POOL is not None:
        _POOL.release()
    _POOL = pool
