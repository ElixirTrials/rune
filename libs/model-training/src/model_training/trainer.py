"""QLoRA training orchestrator.

All GPU-dependent imports (datasets, transformers, trl, torch) are deferred
inside function bodies to ensure CPU-only importability (INFRA-05).

Module-level imports: stdlib only.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path


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
) -> dict[str, object]:
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
        Dict with resolved values for all training parameters.
    """
    resolved: dict[str, object] = {
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


def train_qlora(
    session_id: str,
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
        session_id: Trajectory session ID to train from.
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

    Returns:
        output_dir path where the adapter was saved.

    Raises:
        FileNotFoundError: If the trajectory file does not exist.
        ValueError: If the trajectory is not successful or has no SFT messages.
    """
    # Deferred GPU imports — all in function body for CPU-only importability
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from model_training.peft_utils import build_qlora_config
    from model_training.trajectory import format_for_sft, load_trajectory

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
    resolved_rank: int = params["rank"]  # type: ignore[assignment]
    resolved_alpha: int = params["alpha"]  # type: ignore[assignment]
    resolved_epochs: int = params["epochs"]  # type: ignore[assignment]
    resolved_grad_accum: int = params["grad_accum"]  # type: ignore[assignment]
    resolved_lr_sched: str = params["lr_sched"]  # type: ignore[assignment]
    attn_impl = params["attn_impl"]

    # Resolve model ID — read env var inside function body for monkeypatch testability
    resolved_model_id = params["base_model_id"]
    model_id: str = resolved_model_id or os.environ.get(  # type: ignore[assignment]
        "RUNE_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
    )

    # Load and format trajectory
    trajectory = load_trajectory(session_id)  # raises FileNotFoundError if missing
    messages = format_for_sft(trajectory)
    if not messages:
        raise ValueError(
            f"Trajectory {session_id} is not successful or has no SFT messages"
        )

    # Build dataset
    dataset = Dataset.from_list([{"messages": messages}])

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

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=resolved_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type=resolved_lr_sched,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=resolved_grad_accum,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        eval_strategy="no",
        assistant_only_loss=True,
    )

    # Create and run trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save adapter weights only (PEFT-aware: safetensors + adapter_config.json)
    trainer.save_model(output_dir)

    return output_dir


def train_and_register(
    session_id: str,
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
        model_id = os.environ.get(
            "RUNE_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
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
    )

    # Compute file hash and size from the saved safetensors file
    adapter_file = adapter_dir / "adapter_model.safetensors"
    file_hash = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    file_size_bytes = adapter_file.stat().st_size

    # Build the adapter record
    record = AdapterRecord(
        id=adapter_id,
        version=1,
        task_type=task_type,
        base_model_id=model_id,
        rank=rank if rank is not None else 64,
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
