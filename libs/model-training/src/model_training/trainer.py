"""QLoRA training orchestrator.

All GPU-dependent imports (datasets, transformers, trl, torch) are deferred
inside function bodies to ensure CPU-only importability (INFRA-05).

Module-level imports: stdlib only.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from model_training.training_common import (
    mlflow_log_output_artifacts,
    mlflow_run,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


class _ResolvedParams(TypedDict):
    """Resolved training parameters after merging registry defaults."""

    base_model_id: str | None
    warm_start: str | None
    rank: int
    alpha: int
    epochs: int
    grad_accum: int
    lr_sched: str
    attn_impl: str | None


def _resolve_training_params(
    *,
    model_config_name: str | None,
    base_model_id: str | None,
    warm_start_adapter_id: str | None,
    rank: int | None,
    alpha: int | None,
    epochs: int | None,
    gradient_accumulation_steps: int | None,
    lr_scheduler_type: str | None,
) -> _ResolvedParams:
    """Resolve training parameters from registry defaults and explicit overrides.

    Args:
        model_config_name: Registry lookup key (e.g. "qwen3.5-9b").
        base_model_id: Explicit model ID override.
        warm_start_adapter_id: Explicit warm-start adapter override.
        rank: Explicit LoRA rank override.
        alpha: Explicit LoRA alpha override.
        epochs: Explicit epochs override.
        gradient_accumulation_steps: Explicit grad accum override.
        lr_scheduler_type: Explicit LR scheduler override.

    Returns:
        _ResolvedParams with resolved values for all training parameters.

    Raises:
        KeyError: If model_config_name is not in the registry.
    """
    resolved: _ResolvedParams = {
        "base_model_id": base_model_id,
        "warm_start": warm_start_adapter_id,
        "rank": rank if rank is not None else 64,
        "alpha": alpha if alpha is not None else 128,
        "epochs": epochs if epochs is not None else 3,
        "grad_accum": gradient_accumulation_steps or 16,
        "lr_sched": lr_scheduler_type or "constant",
        "attn_impl": None,
    }

    if model_config_name:
        from model_training.model_configs import ModelRegistry

        mc = ModelRegistry.default().get(model_config_name)
        if base_model_id is None:
            resolved["base_model_id"] = mc.model_id
        if warm_start_adapter_id is None:
            resolved["warm_start"] = mc.warm_start_adapter_id
        if rank is None:
            resolved["rank"] = mc.default_lora_rank
        if alpha is None:
            resolved["alpha"] = mc.default_lora_alpha
        if epochs is None:
            resolved["epochs"] = mc.epochs
        if gradient_accumulation_steps is None:
            resolved["grad_accum"] = mc.gradient_accumulation_steps
        if lr_scheduler_type is None:
            resolved["lr_sched"] = mc.lr_scheduler_type
        resolved["attn_impl"] = mc.attn_implementation

    return resolved


def _override_lora_alpha(model: Any, adapter_name: str, new_alpha: int) -> None:
    """Re-scale a loaded LoRA adapter's effective alpha in place.

    PEFT's ``LoraLayer`` caches ``scaling[adapter_name] = alpha / r`` at
    layer-construction time — mutating ``model.peft_config[adapter].lora_alpha``
    after ``PeftModel.from_pretrained`` does NOT propagate, so we walk
    the module tree and update each layer's cached scaling directly.

    Also mirrors the change into ``peft_config`` so downstream code that
    inspects the config (e.g. saving the adapter) sees the new alpha.
    """
    if hasattr(model, "peft_config") and adapter_name in model.peft_config:
        # Keep config consistent with the module state for any downstream
        # serialization / inspection.
        model.peft_config[adapter_name].lora_alpha = new_alpha

    updated = 0
    for module in model.modules():
        if not hasattr(module, "scaling") or not hasattr(module, "r"):
            continue
        scaling = getattr(module, "scaling", None)
        r_map = getattr(module, "r", None)
        if not isinstance(scaling, dict) or not isinstance(r_map, dict):
            continue
        if adapter_name not in scaling or adapter_name not in r_map:
            continue
        rank = r_map[adapter_name]
        if rank <= 0:
            continue
        scaling[adapter_name] = new_alpha / rank
        updated += 1
    logger.info(
        "LoRA alpha override: adapter=%s new_alpha=%d updated_layers=%d",
        adapter_name,
        new_alpha,
        updated,
    )


def _override_lora_dropout(model: Any, adapter_name: str, new_p: float) -> None:
    """Override the dropout probability of a loaded LoRA adapter in place.

    ``layer.lora_dropout`` is an ``nn.ModuleDict`` keyed by adapter name
    whose values are ``nn.Dropout`` modules. We update the ``.p``
    attribute on each such module so new mini-batches see the new rate.

    Passing ``new_p == 0.0`` effectively disables LoRA dropout for the
    adapter without rewiring the module graph.
    """
    if not 0.0 <= new_p <= 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {new_p}")

    if hasattr(model, "peft_config") and adapter_name in model.peft_config:
        model.peft_config[adapter_name].lora_dropout = new_p

    updated = 0
    for module in model.modules():
        drop_map = getattr(module, "lora_dropout", None)
        if drop_map is None:
            continue
        # lora_dropout is an nn.ModuleDict; tolerate plain dict too.
        if hasattr(drop_map, "__contains__") and adapter_name in drop_map:
            target = drop_map[adapter_name]
            if hasattr(target, "p"):
                target.p = float(new_p)
                updated += 1
    logger.info(
        "LoRA dropout override: adapter=%s new_p=%.3f updated_layers=%d",
        adapter_name,
        new_p,
        updated,
    )


def _validate_data_source(
    session_id: str | None, dataset_path: str | None, encoding_mode: str
) -> None:
    """Enforce mutual exclusivity and encoding_mode whitelist up front.

    Raised early (before any GPU imports) so callers get a fast, cheap
    signal on misconfiguration.
    """
    if dataset_path is not None and session_id is not None:
        raise ValueError("train_qlora: pass dataset_path XOR session_id, not both")
    if dataset_path is None and session_id is None:
        raise ValueError(
            "train_qlora: either dataset_path or session_id must be provided"
        )
    if encoding_mode not in ("multi_turn", "single_turn"):
        raise ValueError(
            "encoding_mode must be 'multi_turn' or 'single_turn', got "
            f"{encoding_mode!r}"
        )


def _build_training_dataset(
    *,
    dataset_cls: Any,
    session_id: str | None,
    dataset_path: str | None,
    encoding_mode: str,
    diff_aware_loss: bool = False,
) -> Any:
    """Build an SFT ``Dataset`` from either a mined-pairs JSONL or a trajectory.

    ``dataset_cls`` is the ``datasets.Dataset`` class injected by the caller
    so this helper stays GPU-import-free at module level while still producing
    a real ``datasets.Dataset`` at call time.

    When ``diff_aware_loss=True`` and ``dataset_path`` is set, ``pre_code`` and
    ``post_code`` columns are attached alongside ``messages`` so the
    :class:`~model_training.diff_loss.DiffWeightedDataCollator` hunk path can
    compute line-level diff weights.  Trajectory-sourced datasets do not carry
    pre/post context, so the collator will log-warn-once and fall back to the
    legacy set-based path.
    """
    from typing import Literal, cast  # noqa: PLC0415

    from model_training.d2l_data import (  # type: ignore[attr-defined]  # noqa: PLC0415
        load_jsonl,  # noqa: PLC0415
        pairs_to_chat_messages,
    )
    from model_training.trajectory import (  # noqa: PLC0415
        format_for_sft,
        load_trajectory,
    )

    if dataset_path is not None:
        pairs = load_jsonl(dataset_path)
        mode = cast(Literal["multi_turn", "single_turn"], encoding_mode)
        conversations, pre_post = pairs_to_chat_messages(pairs, mode=mode)
        if not conversations:
            raise ValueError(
                f"dataset_path {dataset_path} produced no SFT conversations"
            )
        if diff_aware_loss:
            return dataset_cls.from_list(
                [
                    {
                        "messages": c,
                        "pre_code": pp["pre_code"],
                        "post_code": pp["post_code"],
                    }
                    for c, pp in zip(conversations, pre_post)
                ]
            )
        return dataset_cls.from_list([{"messages": c} for c in conversations])

    # session_id is not None at this point (validated above).
    trajectory = load_trajectory(str(session_id))
    messages = format_for_sft(trajectory)
    if not messages:
        raise ValueError(
            f"Trajectory {session_id} is not successful or has no SFT messages"
        )
    return dataset_cls.from_list([{"messages": messages}])


def _construct_sft_trainer(
    *,
    sft_trainer_cls: Any,
    model: Any,
    args: Any,
    dataset: Any,
    tokenizer: Any,
    lora_config: Any,
    diff_aware_loss: bool,
    diff_changed_weight: float,
    diff_unchanged_weight: float,
) -> Any:
    """Pick between vanilla SFTTrainer and the diff-aware subclass.

    Isolated so ``train_qlora`` stays under the ``C901`` complexity
    threshold. When ``diff_aware_loss=True``, builds a
    ``DiffAwareSFTTrainer`` and wraps its auto-built collator with
    :class:`~model_training.diff_loss.DiffWeightedDataCollator` so each
    batch carries a ``loss_weights`` tensor.
    """
    if not diff_aware_loss:
        return sft_trainer_cls(
            model=model,
            args=args,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
        )

    from model_training.diff_loss import (  # noqa: PLC0415
        build_diff_aware_sft_trainer,
    )

    return build_diff_aware_sft_trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        changed_weight=diff_changed_weight,
        unchanged_weight=diff_unchanged_weight,
        tokenizer=tokenizer,
    )


def _build_sft_config(
    *,
    sft_config_cls: Any,
    output_dir: str,
    resolved_epochs: int,
    learning_rate: float,
    warmup_ratio: float | None,
    resolved_lr_sched: str,
    resolved_grad_accum: int,
    report_to: str,
    diff_aware_loss: bool,
    neftune_noise_alpha: float | None,
) -> Any:
    """Construct an SFTConfig with optional NEFTune support.

    Extracted to keep ``train_qlora`` under the C901 complexity limit.
    ``neftune_noise_alpha`` is only passed when not None so SFTConfig's own
    default (feature disabled) applies when the caller omits it.
    """
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": resolved_epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio if warmup_ratio is not None else 0.03,
        "lr_scheduler_type": resolved_lr_sched,
        "bf16": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": resolved_grad_accum,
        "save_strategy": "no",
        "logging_steps": 1,
        "report_to": report_to,
        "eval_strategy": "no",
        "assistant_only_loss": not diff_aware_loss,
    }
    if neftune_noise_alpha is not None:
        kwargs["neftune_noise_alpha"] = neftune_noise_alpha
    if diff_aware_loss:
        # TRL strips unknown columns by default; keep pre_code / post_code
        # so the hunk-path collator can see them.
        kwargs["remove_unused_columns"] = False
    return sft_config_cls(**kwargs)


def train_qlora(
    session_id: str | None,
    adapter_id: str,
    output_dir: str,
    *,
    base_model_id: str | None = None,
    task_type: str = "code-gen",
    rank: int | None = None,
    alpha: int | None = None,
    epochs: int | None = None,
    learning_rate: float = 2e-4,
    model_config_name: str | None = None,
    warm_start_adapter_id: str | None = None,
    gradient_accumulation_steps: int | None = None,
    lr_scheduler_type: str | None = None,
    mlflow_experiment: str = "rune-qlora",
    mlflow_tracking_uri: str | None = None,
    dataset_path: str | None = None,
    encoding_mode: str = "multi_turn",
    diff_aware_loss: bool = False,
    diff_changed_weight: float = 1.0,
    diff_unchanged_weight: float = 0.3,
    override_lora_alpha: int | None = None,
    override_lora_dropout: float | None = None,
    warmup_ratio: float | None = None,
    neftune_noise_alpha: float | None = None,
) -> str:
    """Train a QLoRA adapter from a recorded coding trajectory.

    Orchestrates the full training pipeline: load trajectory, format as SFT
    messages, build dataset, load model with NF4 quantization, train with SFT,
    and save the adapter to output_dir.

    When model_config_name is provided, training defaults (rank, alpha, LR
    schedule, gradient accumulation, attention implementation) are loaded from
    the model registry. When the registry config specifies a warm_start_adapter_id,
    the pre-trained adapter is loaded via PeftModel.from_pretrained and training
    continues from those weights instead of initializing fresh LoRA matrices.

    All GPU imports are deferred to this function body; the module is safe to
    import in CPU-only environments.

    Args:
        session_id: Trajectory session ID to train from. Mutually exclusive
            with dataset_path.
        adapter_id: Unique identifier for the resulting adapter.
        output_dir: Directory to save the trained adapter weights.
        base_model_id: HuggingFace model ID. Overrides registry default.
        task_type: Task category (e.g. 'code-gen', 'bug-fix').
        rank: LoRA rank. Defaults to registry value or 64.
        alpha: LoRA alpha scaling factor. Defaults to registry value or 128.
        epochs: Number of training epochs. Defaults to registry value or 3.
        learning_rate: Optimizer learning rate.
        model_config_name: Registry lookup key (e.g. "qwen3.5-9b"). When set,
            populates defaults from the model registry.
        warm_start_adapter_id: Pre-trained PEFT adapter to continue from.
            Overrides the registry's warm_start_adapter_id.
        gradient_accumulation_steps: Gradient accumulation steps. Defaults to
            registry value or 16.
        lr_scheduler_type: LR scheduler type. Defaults to registry value or
            "constant".
        mlflow_experiment: MLflow experiment name. Set RUNE_DISABLE_MLFLOW=1
            in the env to disable MLflow tracking entirely.
        mlflow_tracking_uri: Optional MLflow tracking URI override. Falls back
            to the MLFLOW_TRACKING_URI env var, then to "./mlruns".
        dataset_path: Path to a JSONL file of mined pair records (as produced
            by ``scripts/mine_github.py --batch``). Mutually exclusive with
            session_id; when set, the training dataset is built by
            ``pairs_to_chat_messages`` instead of ``format_for_sft``.
        encoding_mode: ``"multi_turn"`` clusters pairs by source task and
            emits one conversation per task; ``"single_turn"`` emits one
            conversation per pair. Only used when ``dataset_path`` is set.
        diff_aware_loss: When True, wrap the SFT collator with
            ``DiffWeightedDataCollator`` and swap in ``DiffAwareSFTTrainer``
            so per-token loss is scaled by the diff between assistant
            output and the masked user/context span. Disables
            ``assistant_only_loss`` on SFTConfig since the custom collator
            provides equivalent (stricter) masking.
        diff_changed_weight: Per-token weight applied to assistant tokens
            absent from the masked context span. Only used when
            ``diff_aware_loss=True``. Defaults to ``1.0``.
        diff_unchanged_weight: Per-token weight applied to assistant
            tokens also present in the masked context span. Only used
            when ``diff_aware_loss=True``. Defaults to ``0.3``.
        override_lora_alpha: Post-load override for the warm-start
            adapter's effective alpha. Applied via a module-tree walk
            after ``PeftModel.from_pretrained`` since PEFT caches
            ``scaling = alpha / r`` per layer. Has no effect when no
            warm-start is in use.
        override_lora_dropout: Post-load override for the warm-start
            adapter's LoRA dropout probability. Applied the same way as
            ``override_lora_alpha`` — by walking the module tree and
            mutating each layer's ``lora_dropout[adapter_name].p``.
        warmup_ratio: Linear warmup fraction of total training steps passed
            to ``SFTConfig``. Defaults to ``0.03`` when ``None``.
        neftune_noise_alpha: NEFTune noise alpha for embedding perturbation.
            When set, passed to ``SFTConfig(neftune_noise_alpha=...)``.
            When ``None`` (default), the kwarg is omitted so SFTConfig's
            own default (feature disabled) applies.

    Returns:
        output_dir path where the adapter was saved.

    Raises:
        FileNotFoundError: If the trajectory file does not exist.
        ValueError: If the trajectory is not successful or has no SFT messages.
    """
    # Validate mutually-exclusive data sources up front (no GPU imports yet).
    _validate_data_source(session_id, dataset_path, encoding_mode)

    # Deferred GPU imports — all in function body for CPU-only importability
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from model_training.peft_utils import build_qlora_config

    # Resolve defaults from model registry and explicit overrides
    params = _resolve_training_params(
        model_config_name=model_config_name,
        base_model_id=base_model_id,
        warm_start_adapter_id=warm_start_adapter_id,
        rank=rank,
        alpha=alpha,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
    )
    warm_start = params["warm_start"]
    resolved_rank = params["rank"]
    resolved_alpha = params["alpha"]
    resolved_epochs = params["epochs"]
    resolved_grad_accum = params["grad_accum"]
    resolved_lr_sched = params["lr_sched"]
    attn_impl = params["attn_impl"]

    # Resolve model ID — read env var inside function body for monkeypatch testability
    resolved_model_id = params["base_model_id"]
    model_id: str = resolved_model_id or os.environ.get(
        "RUNE_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
    )  # type: ignore[assignment]

    # Build dataset from either a mined-pairs JSONL or a recorded trajectory.
    dataset = _build_training_dataset(
        dataset_cls=Dataset,
        session_id=session_id,
        dataset_path=dataset_path,
        encoding_mode=encoding_mode,
        diff_aware_loss=diff_aware_loss,
    )

    # NF4 quantization config (bfloat16 compute dtype prevents silent NaN loss)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    model_kwargs: dict[str, object] = {
        "quantization_config": bnb_config,
        "device_map": "auto",
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Warm-start: load pre-trained adapter and enable gradient on LoRA params
    lora_config = None
    if warm_start:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(warm_start))  # type: ignore[assignment]
        # Enable gradient on all adapter parameters for continued training
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

        # Post-load overrides. The saved adapter's rank and target_modules
        # are locked by the safetensor shapes, but alpha and dropout are
        # runtime quantities and can be retuned by HPO without discarding
        # the warm-start.
        adapter_name = getattr(model, "active_adapter", None) or "default"
        if override_lora_alpha is not None:
            _override_lora_alpha(model, adapter_name, override_lora_alpha)
        if override_lora_dropout is not None:
            _override_lora_dropout(model, adapter_name, override_lora_dropout)
    else:
        # Fresh LoRA initialization — try probe cache for target modules
        target_modules = ["q_proj", "v_proj"]
        if model_config_name:
            from model_training.d2l_probe import load_probe_cache

            cache = load_probe_cache(model_config_name)
            if cache and "target_modules" in cache:
                target_modules = cache["target_modules"]

        lora_config = build_qlora_config(
            rank=resolved_rank,
            alpha=resolved_alpha,
            target_modules=target_modules,
            dropout=0.1,
        )

    # Configure MLflow (silent no-op when disabled via env or mlflow missing)
    mlflow_enabled = setup_mlflow(
        experiment_name=mlflow_experiment, tracking_uri=mlflow_tracking_uri
    )
    report_to = "mlflow" if mlflow_enabled else "none"

    # Training arguments. When diff_aware_loss is on, our custom collator
    # subsumes the assistant-only masking, so we disable SFTConfig's
    # own flag to avoid double-masking.
    training_args = _build_sft_config(
        sft_config_cls=SFTConfig,
        output_dir=output_dir,
        resolved_epochs=resolved_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        resolved_lr_sched=resolved_lr_sched,
        resolved_grad_accum=resolved_grad_accum,
        report_to=report_to,
        diff_aware_loss=diff_aware_loss,
        neftune_noise_alpha=neftune_noise_alpha,
    )

    trainer = _construct_sft_trainer(
        sft_trainer_cls=SFTTrainer,
        model=model,
        args=training_args,
        dataset=dataset,
        tokenizer=tokenizer,
        lora_config=lora_config,
        diff_aware_loss=diff_aware_loss,
        diff_changed_weight=diff_changed_weight,
        diff_unchanged_weight=diff_unchanged_weight,
    )

    # Log training-run metadata (params streamed as per-step metrics by TRL's
    # MLflow reporter via training_args.report_to="mlflow"). When disabled,
    # contextlib.nullcontext is a zero-cost placeholder.
    run_params = {
        "model_id": model_id,
        "warm_start": warm_start or "",
        "rank": resolved_rank,
        "alpha": resolved_alpha,
        "epochs": resolved_epochs,
        "learning_rate": learning_rate,
        "grad_accum": resolved_grad_accum,
        "lr_scheduler_type": resolved_lr_sched,
        "attn_implementation": attn_impl or "",
        "dataset_size": len(dataset),
        "assistant_only_loss": True,
        "task_type": task_type,
        "adapter_id": adapter_id,
        "session_id": session_id or "",
        "dataset_path": dataset_path or "",
        "encoding_mode": encoding_mode,
        "diff_aware_loss": diff_aware_loss,
        "diff_changed_weight": diff_changed_weight if diff_aware_loss else "",
        "diff_unchanged_weight": diff_unchanged_weight if diff_aware_loss else "",
        "override_lora_alpha": (
            override_lora_alpha if override_lora_alpha is not None else ""
        ),
        "override_lora_dropout": (
            override_lora_dropout if override_lora_dropout is not None else ""
        ),
        "warmup_ratio": warmup_ratio if warmup_ratio is not None else 0.03,
        "neftune_noise_alpha": (
            neftune_noise_alpha if neftune_noise_alpha is not None else ""
        ),
    }
    with mlflow_run(
        enabled=mlflow_enabled,
        run_name=f"{adapter_id}-r{resolved_rank}-lr{learning_rate:.1e}",
        params=run_params,
    ):
        trainer.train()

        # Save adapter weights only (PEFT-aware: safetensors + adapter_config.json)
        trainer.save_model(output_dir)

        if mlflow_enabled:
            mlflow_log_output_artifacts(output_dir)

    return output_dir


def train_and_register(
    session_id: str | None,
    adapter_id: str,
    *,
    base_model_id: str | None = None,
    task_type: str = "code-gen",
    rank: int | None = None,
    alpha: int | None = None,
    epochs: int | None = None,
    learning_rate: float = 2e-4,
    model_config_name: str | None = None,
    warm_start_adapter_id: str | None = None,
    gradient_accumulation_steps: int | None = None,
    lr_scheduler_type: str | None = None,
    database_url: str | None = None,
    mlflow_experiment: str = "rune-qlora",
    mlflow_tracking_uri: str | None = None,
    dataset_path: str | None = None,
    encoding_mode: str = "multi_turn",
    diff_aware_loss: bool = False,
    diff_changed_weight: float = 1.0,
    diff_unchanged_weight: float = 0.3,
    override_lora_alpha: int | None = None,
    override_lora_dropout: float | None = None,
    warmup_ratio: float | None = None,
    neftune_noise_alpha: float | None = None,
) -> str:
    """Train a QLoRA adapter and register it in the AdapterRegistry.

    Combines train_qlora() with AdapterRegistry.store() to produce a fully
    registered adapter ready for vLLM serving.

    Args:
        session_id: Trajectory session ID to train from.
        adapter_id: Unique identifier for the resulting adapter.
        base_model_id: HuggingFace model ID. Defaults to RUNE_BASE_MODEL env var.
        task_type: Task category (e.g. 'code-gen', 'bug-fix').
        rank: LoRA rank. Defaults to registry value or 64.
        alpha: LoRA alpha scaling factor. Defaults to registry value or 128.
        epochs: Number of training epochs. Defaults to registry value or 3.
        learning_rate: Optimizer learning rate.
        model_config_name: Registry lookup key (e.g. "qwen3.5-9b").
        warm_start_adapter_id: Pre-trained PEFT adapter to continue from.
        gradient_accumulation_steps: Gradient accumulation steps.
        lr_scheduler_type: LR scheduler type.
        database_url: SQLAlchemy database URL. Defaults to RUNE_DATABASE_URL env var
            or "sqlite:///{home}/.rune/rune.db".
        mlflow_experiment: MLflow experiment name. Set RUNE_DISABLE_MLFLOW=1 to
            skip MLflow tracking.
        mlflow_tracking_uri: Optional MLflow tracking URI override.
        dataset_path: JSONL of mined pair records; mutually exclusive with
            session_id.
        encoding_mode: ``"multi_turn"`` or ``"single_turn"`` for pair→chat
            conversion.
        diff_aware_loss: Enable diff-aware per-token loss weighting.
        diff_changed_weight: Weight for assistant tokens not in the masked
            context span when ``diff_aware_loss=True``.
        diff_unchanged_weight: Weight for assistant tokens present in the
            masked context span when ``diff_aware_loss=True``.
        override_lora_alpha: Post-load override for the warm-start
            adapter's effective alpha (module-tree walk).
        override_lora_dropout: Post-load override for the warm-start
            adapter's LoRA dropout probability.
        warmup_ratio: Linear warmup fraction forwarded to ``train_qlora``.
            Defaults to ``0.03`` when ``None``.
        neftune_noise_alpha: NEFTune noise alpha forwarded to ``train_qlora``.
            When ``None`` (default), NEFTune is disabled.

    Returns:
        adapter_id of the registered adapter.

    Raises:
        FileNotFoundError: If the trajectory file does not exist.
        ValueError: If the trajectory is not successful or has no SFT messages.
    """
    from adapter_registry.models import AdapterRecord
    from adapter_registry.registry import AdapterRegistry
    from sqlalchemy import create_engine

    # Resolve adapter output dir — read env var inside function body
    adapter_base = os.environ.get("RUNE_ADAPTER_DIR")
    if adapter_base:
        adapter_dir = Path(adapter_base) / adapter_id
    else:
        adapter_dir = Path.home() / ".rune" / "adapters" / adapter_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model ID inside function body for testability
    model_id = base_model_id
    if model_id is None and model_config_name:
        from model_training.model_configs import ModelRegistry

        model_id = ModelRegistry.default().get(model_config_name).model_id
    if model_id is None:
        model_id = os.environ.get("RUNE_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

    # Resolve training params to get the actual rank used for training
    resolved = _resolve_training_params(
        model_config_name=model_config_name,
        base_model_id=base_model_id,
        warm_start_adapter_id=warm_start_adapter_id,
        rank=rank,
        alpha=alpha,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
    )

    output_dir = str(adapter_dir)
    train_qlora(
        session_id=session_id,
        adapter_id=adapter_id,
        output_dir=output_dir,
        base_model_id=model_id,
        task_type=task_type,
        rank=rank,
        alpha=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        model_config_name=model_config_name,
        warm_start_adapter_id=warm_start_adapter_id,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
        dataset_path=dataset_path,
        encoding_mode=encoding_mode,
        diff_aware_loss=diff_aware_loss,
        diff_changed_weight=diff_changed_weight,
        diff_unchanged_weight=diff_unchanged_weight,
        override_lora_alpha=override_lora_alpha,
        override_lora_dropout=override_lora_dropout,
        warmup_ratio=warmup_ratio,
        neftune_noise_alpha=neftune_noise_alpha,
    )

    # Compute file hash and size from the saved safetensors file
    adapter_file = adapter_dir / "adapter_model.safetensors"
    file_hash = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    file_size_bytes = adapter_file.stat().st_size

    # Build the adapter record with the registry-resolved rank
    record = AdapterRecord(
        id=adapter_id,
        version=1,
        task_type=task_type,
        base_model_id=model_id,
        rank=resolved["rank"],
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        file_path=output_dir,
        file_hash=file_hash,
        file_size_bytes=file_size_bytes,
        source="qlora",
        session_id=session_id,
    )

    # Resolve DB URL inside function body for testability
    db_url = database_url or os.environ.get(
        "RUNE_DATABASE_URL",
        f"sqlite:///{Path.home() / '.rune' / 'rune.db'}",
    )
    engine = create_engine(db_url)
    registry = AdapterRegistry(engine=engine)
    registry.store(record)

    return adapter_id
