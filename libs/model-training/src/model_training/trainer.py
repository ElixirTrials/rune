"""QLoRA training orchestrator.

All GPU-dependent imports (datasets, transformers, trl, torch) are deferred
inside function bodies to ensure CPU-only importability (INFRA-05).

Module-level imports: stdlib only.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


def _setup_mlflow_trainer(
    experiment_name: str, tracking_uri: str | None
) -> bool:
    """Configure MLflow for QLoRA training runs.

    Returns True when MLflow is usable and configured; False when tracking
    should be skipped. Skipping happens when RUNE_DISABLE_MLFLOW=1 is set
    in the environment, or when mlflow itself is not importable.

    Tracking URI precedence: explicit ``tracking_uri`` arg, then the
    ``MLFLOW_TRACKING_URI`` env var, then ``./mlruns`` as a local-dev fallback.
    Mirrors the pattern in ``d2l_train._setup_mlflow``.
    """
    if os.environ.get("RUNE_DISABLE_MLFLOW") == "1":
        return False
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        return False
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow enabled: tracking_uri=%s experiment=%s", uri, experiment_name
    )
    return True


def _mlflow_log_params(params: dict[str, Any]) -> None:
    """Log a dict of params to the active MLflow run. Silent no-op on failure."""
    try:
        import mlflow  # noqa: PLC0415

        mlflow.log_params(params)
    except Exception:  # noqa: BLE001 — logging must never break training
        logger.debug("mlflow.log_params skipped", exc_info=True)


def _mlflow_log_artifact(path: str) -> None:
    """Log a file artifact to the active MLflow run. Silent no-op on failure."""
    try:
        import mlflow  # noqa: PLC0415

        mlflow.log_artifact(path)
    except Exception:  # noqa: BLE001
        logger.debug("mlflow.log_artifact skipped for %s", path, exc_info=True)


def _mlflow_log_output_artifacts(output_dir: str) -> None:
    """Log the saved adapter's safetensors + config.json to MLflow, if present."""
    adapter_safetensors = Path(output_dir) / "adapter_model.safetensors"
    adapter_config = Path(output_dir) / "adapter_config.json"
    if adapter_safetensors.exists():
        _mlflow_log_artifact(str(adapter_safetensors))
    if adapter_config.exists():
        _mlflow_log_artifact(str(adapter_config))


@contextmanager
def _mlflow_run(
    *, enabled: bool, run_name: str, params: dict[str, Any]
) -> Iterator[None]:
    """Context manager that starts an MLflow run when enabled, else no-ops.

    When enabled, logs ``params`` at entry and ensures ``mlflow.end_run()`` on
    exit even if training raises. When disabled, the body runs unchanged.
    """
    if not enabled:
        yield
        return
    import mlflow  # noqa: PLC0415

    mlflow.start_run(run_name=run_name)
    try:
        _mlflow_log_params(params)
        yield
    finally:
        mlflow.end_run()


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


def _validate_data_source(
    session_id: str | None, dataset_path: str | None, encoding_mode: str
) -> None:
    """Enforce mutual exclusivity and encoding_mode whitelist up front.

    Raised early (before any GPU imports) so callers get a fast, cheap
    signal on misconfiguration.
    """
    if dataset_path is not None and session_id is not None:
        raise ValueError(
            "train_qlora: pass dataset_path XOR session_id, not both"
        )
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
) -> Any:
    """Build an SFT ``Dataset`` from either a mined-pairs JSONL or a trajectory.

    ``dataset_cls`` is the ``datasets.Dataset`` class injected by the caller
    so this helper stays GPU-import-free at module level while still producing
    a real ``datasets.Dataset`` at call time.
    """
    from typing import Literal, cast  # noqa: PLC0415

    from model_training.d2l_data import (  # noqa: PLC0415
        load_jsonl,
        pairs_to_chat_messages,
    )
    from model_training.trajectory import (  # noqa: PLC0415
        format_for_sft,
        load_trajectory,
    )

    if dataset_path is not None:
        pairs = load_jsonl(dataset_path)
        mode = cast(Literal["multi_turn", "single_turn"], encoding_mode)
        conversations = pairs_to_chat_messages(pairs, mode=mode)
        if not conversations:
            raise ValueError(
                f"dataset_path {dataset_path} produced no SFT conversations"
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
    mlflow_enabled = _setup_mlflow_trainer(
        experiment_name=mlflow_experiment, tracking_uri=mlflow_tracking_uri
    )
    report_to = "mlflow" if mlflow_enabled else "none"

    # Training arguments. When diff_aware_loss is on, our custom collator
    # subsumes the assistant-only masking, so we disable SFTConfig's
    # own flag to avoid double-masking.
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
        report_to=report_to,
        eval_strategy="no",
        assistant_only_loss=not diff_aware_loss,
    )

    # Create the trainer. Vanilla path stays intact for default runs;
    # diff-aware path builds the subclassed trainer and wraps its
    # auto-built collator so loss_weights are produced per batch.
    if diff_aware_loss:
        from model_training.diff_loss import (  # noqa: PLC0415
            DiffWeightedDataCollator,
            build_diff_aware_sft_trainer,
        )

        trainer = build_diff_aware_sft_trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
            changed_weight=diff_changed_weight,
            unchanged_weight=diff_unchanged_weight,
        )
        trainer.data_collator = DiffWeightedDataCollator(
            trainer.data_collator,
            changed_weight=diff_changed_weight,
            unchanged_weight=diff_unchanged_weight,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
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
    }
    with _mlflow_run(
        enabled=mlflow_enabled,
        run_name=f"{adapter_id}-r{resolved_rank}-lr{learning_rate:.1e}",
        params=run_params,
    ):
        trainer.train()

        # Save adapter weights only (PEFT-aware: safetensors + adapter_config.json)
        trainer.save_model(output_dir)

        if mlflow_enabled:
            _mlflow_log_output_artifacts(output_dir)

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
