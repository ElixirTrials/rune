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


def train_qlora(
    session_id: str,
    adapter_id: str,
    output_dir: str,
    *,
    base_model_id: str | None = None,
    task_type: str = "code-gen",
    rank: int = 64,
    alpha: int = 128,
    epochs: int = 3,
    learning_rate: float = 2e-4,
) -> str:
    """Train a QLoRA adapter from a recorded coding trajectory.

    Orchestrates the full training pipeline: load trajectory, format as SFT
    messages, build dataset, load model with NF4 quantization, train with SFT,
    and save the adapter to output_dir.

    All GPU imports are deferred to this function body; the module is safe to
    import in CPU-only environments.

    Args:
        session_id: Trajectory session ID to train from.
        adapter_id: Unique identifier for the resulting adapter.
        output_dir: Directory to save the trained adapter weights.
        base_model_id: HuggingFace model ID. Defaults to RUNE_BASE_MODEL env var
            or "Qwen/Qwen2.5-Coder-7B-Instruct".
        task_type: Task category (e.g. 'code-gen', 'bug-fix').
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.

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

    # Resolve model ID — read env var inside function body for monkeypatch testability
    model_id = base_model_id or os.environ.get(
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build LoRA config (bfloat16 is set in BitsAndBytesConfig, not LoraConfig)
    lora_config = build_qlora_config(
        rank=rank,
        alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        dropout=0.1,
    )

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        eval_strategy="no",
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
    rank: int = 64,
    alpha: int = 128,
    epochs: int = 3,
    learning_rate: float = 2e-4,
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
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
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
    model_id = base_model_id or os.environ.get(
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
        rank=rank,
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
