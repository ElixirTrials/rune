"""Round-2 hypernetwork training: per-bin oracle teacher distillation.

This module mirrors :mod:`model_training.d2l_train`'s two-pass teacher/student
step but replaces the bare-base-model teacher forward with a per-record
oracle-LoRA teacher forward. The oracle is applied via the SAME
``apply_functional_lora`` context manager used for the student pass, which
monkey-patches the base model's weight tensors in place and restores on
exit. The base model is never structurally mutated (no ``PeftModel``
wrappers, no ``LoraLayer`` replacements), so there is no possibility of
hook leakage between teacher and student passes.

When a record's bin has no registered oracle and ``oracle_fallback='skip'``
(the default), the record is dropped upstream in ``_training_step_round2``
and the teacher-forward helper is not invoked. When
``oracle_fallback='base_model'`` (ablation only), the helper is invoked
with ``oracle_lora_dict=None`` and falls through to the bare base model.

GPU imports are deferred inside function bodies per INFRA-05 so this
module stays importable in CPU-only CI.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin wrapper around apply_functional_lora so tests can monkeypatch it.
# ---------------------------------------------------------------------------


def _apply_functional_lora(base_model: Any, lora_dict: Any, hc: Any) -> Any:
    """Sole injection point for functional LoRA in the round-2 training path.

    Tests monkeypatch this module-level name to verify routing without GPU
    tensors (see :mod:`test_round2_train`). Do NOT inline the
    ``apply_functional_lora`` import at call sites — that would silently
    break testability by routing around the monkeypatch seam.

    Args:
        base_model: Base language model.
        lora_dict: Functional-LoRA tensors (``{module: {A, B}}``).
        hc: HypernetConfig passed through to ``apply_functional_lora``.

    Returns:
        Context manager from :func:`model_training.d2l_lora.apply_functional_lora`
        that monkey-patches base_model weights on enter and restores on exit.
    """
    from model_training.d2l_lora import apply_functional_lora  # noqa: PLC0415

    return apply_functional_lora(base_model, lora_dict, hc)


def _teacher_forward_with_oracle(
    *,
    base_model: Any,
    oracle_lora_dict: Any | None,
    hc: Any,
    inputs: dict[str, Any],
) -> Any:
    """Run the teacher forward pass and return logits.

    When ``oracle_lora_dict`` is not ``None``, the oracle's functional-LoRA
    tensors are applied to the base model via ``apply_functional_lora`` for
    the duration of the forward pass and reverted on exit. When ``None``,
    the bare base model is used (ablation fallback for ``oracle_fallback=
    'base_model'``).

    Note: this helper does NOT wrap the call in ``torch.no_grad()``; the
    caller owns the no-grad context. Keeping this helper free of torch
    imports lets CPU-only tests exercise the routing logic with MagicMocks.

    Args:
        base_model: Base language model (HF ``AutoModelForCausalLM``).
        oracle_lora_dict: Functional-LoRA tensors from
            :class:`OracleAdapterCache` (``{module: {A, B}}``) or ``None``.
        hc: HypernetConfig (passed through to ``apply_functional_lora``).
        inputs: Tokenized inputs dict with ``input_ids`` + ``attention_mask``.

    Returns:
        The logits tensor from the chosen teacher.
    """
    if oracle_lora_dict is None:
        out = base_model(**inputs, output_hidden_states=False)
        return out.logits
    with _apply_functional_lora(base_model, oracle_lora_dict, hc):
        out = base_model(**inputs, output_hidden_states=False)
    return out.logits


# -----------------------------------------------------------------------------
# Round-1 internals reused here. Bound to module-scope names so tests can
# monkeypatch them cleanly. _apply_functional_lora is already defined above;
# it is shared between teacher and student passes.
# -----------------------------------------------------------------------------


def _extract_activations_with_model(**kwargs: Any) -> Any:
    """Thin wrapper so tests can monkeypatch activation extraction."""
    from model_training.d2l_probe import (  # noqa: PLC0415
        extract_activations_with_model,
    )

    return extract_activations_with_model(**kwargs)


def _compute_kl_ce_loss(*args: Any, **kwargs: Any) -> Any:
    """Thin wrapper so tests can monkeypatch the loss function."""
    from model_training.d2l_train import _compute_kl_ce_loss as _impl  # noqa: PLC0415

    return _impl(*args, **kwargs)


def _torch_no_grad() -> Any:
    """Thin wrapper so tests can monkeypatch torch.no_grad()."""
    import torch  # noqa: PLC0415

    return torch.no_grad()


def _training_step_round2(
    *,
    record: dict[str, Any],
    base_model: Any,
    tokenizer: Any,
    hypernet: Any,
    hc: Any,
    config: Any,
    oracle_cache: Any,
) -> tuple[Any, dict[str, float]]:
    """Single round-2 training step.

    Mirrors :func:`model_training.d2l_train._training_step` but replaces the
    teacher forward pass with a per-record oracle-LoRA forward, applied via
    the same :func:`apply_functional_lora` mechanism used for the student
    pass. When the record's bin has no registered oracle:

    - ``config.oracle_fallback == "skip"`` (default) → returns ``(None, {})``
      so the caller advances without an optimizer step.
    - ``config.oracle_fallback == "base_model"`` (ablation) → teacher runs
      against the bare base model (identical to round-1).

    Args:
        record: Training record (JSONL row).
        base_model: Base LM in eval mode.
        tokenizer: Tokenizer matching base_model.
        hypernet: HyperLoRA in train mode.
        hc: HypernetConfig with layer_indices + lora_config.
        config: :class:`Round2TrainConfig` instance.
        oracle_cache: :class:`OracleAdapterCache` instance.

    Returns:
        ``(loss_tensor, metrics_dict)`` or ``(None, {})`` when skipped.
    """
    from model_training.oracle_cache import _bin_key_for_record  # noqa: PLC0415

    bin_key = _bin_key_for_record(record)
    oracle_lora_dict = oracle_cache.get(bin_key)

    if oracle_lora_dict is None and config.oracle_fallback == "skip":
        logger.info("No oracle for bin %r; skipping record per config", bin_key)
        return (None, {})

    # --- Pass 1: activation extraction (activation_text only) ---
    features, attn_mask = _extract_activations_with_model(
        text=record["activation_text"],
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=list(hc.layer_indices),
        max_length=config.max_length,
    )

    # --- Hypernet forward (keeps autograd graph) ---
    hypernet_lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)

    # --- answer_start (token offset where the answer begins in teacher_text) ---
    answer_start = len(
        tokenizer(
            record["activation_text"],
            truncation=True,
            max_length=config.max_length,
        )["input_ids"]
    )

    # --- Tokenize teacher_text for the teacher + student passes ---
    teacher_inputs = tokenizer(
        record["teacher_text"],
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
    )
    try:
        device = next(base_model.parameters()).device
    except StopIteration:
        import torch  # noqa: PLC0415

        device = torch.device("cpu")
    teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

    # --- Pass 2 teacher: oracle LoRA (or bare base fallback) under no_grad ---
    with _torch_no_grad():
        teacher_logits = _teacher_forward_with_oracle(
            base_model=base_model,
            oracle_lora_dict=oracle_lora_dict,
            hc=hc,
            inputs=teacher_inputs,
        )

    # --- Pass 2 student: base model with hypernet functional-LoRA patches ---
    with _apply_functional_lora(base_model, hypernet_lora_dict, hc):
        student_out = base_model(**teacher_inputs, output_hidden_states=False)
    student_logits = student_out.logits

    return _compute_kl_ce_loss(student_logits, teacher_logits, answer_start, config)


# -----------------------------------------------------------------------------
# Infrastructure wrappers (monkeypatch seams for unit tests)
# -----------------------------------------------------------------------------


def _load_records(dataset_path: str) -> list[dict[str, Any]]:
    """Load a JSONL manifest into a list of records."""
    from model_training.d2l_data import load_jsonl  # noqa: PLC0415

    return list(load_jsonl(dataset_path))


def _open_registry(url: str) -> Any:
    """Open an AdapterRegistry for the given SQLAlchemy URL."""
    from adapter_registry.registry import AdapterRegistry  # noqa: PLC0415
    from sqlmodel import create_engine  # noqa: PLC0415

    engine = create_engine(url)
    return AdapterRegistry(engine=engine)


def _setup_training(config: Any) -> dict[str, Any]:
    """Load base model, tokenizer, hypernet, hc.

    Mirrors the inline setup block in :func:`model_training.d2l_train.train_d2l_qwen3`
    (lines 563–594 as of the d710005 baseline). Duplicating the block keeps
    round-1 untouched; a future refactor can factor these into shared helpers.
    Returns a dict of handles consumed by :func:`_run_training_loop`.
    """
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.hypernet import HyperLoRA  # noqa: PLC0415
    from shared.hardware import get_best_device  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from model_training.d2l_config import build_hypernet_config  # noqa: PLC0415
    from model_training.d2l_train import _require_probe_cache  # noqa: PLC0415
    from model_training.sakana_d2l import (  # noqa: PLC0415
        get_aggregator_config,
        transfer_aggregator_weights,
    )


    if not config.smoke_test:
        _require_probe_cache(config.model_config_name)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        output_hidden_states=True,
    ).eval()

    hc = build_hypernet_config(
        config.model_config_name,
        aggregator_config=get_aggregator_config(config.sakana_checkpoint_path),
        lora_r=config.lora_r,
    )
    hypernet = HyperLoRA(hc).to(torch.float32)
    hypernet = transfer_aggregator_weights(hypernet, config.sakana_checkpoint_path)
    hypernet.train()

    device = torch.device(get_best_device())
    logger.info("Round-2 using device: %s", device)
    base_model = base_model.to(device)
    hypernet = hypernet.to(device)

    return {
        "base_model": base_model,
        "tokenizer": tokenizer,
        "hypernet": hypernet,
        "hc": hc,
        "device": device,
    }


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def train_d2l_qwen3_round2(
    config: Any,
    *,
    kill_switch_evaluate_fn: Any | None = None,
) -> dict[str, Any]:
    """Round-2 hypernetwork training loop with per-bin oracle teachers.

    Flow:
      1. Load records from ``config.dataset_path``.
      2. Open the AdapterRegistry, audit oracle coverage, abort when
         coverage < ``config.min_oracle_coverage``.
      3. When ``config.dry_run``: return a report dict and exit.
      4. Setup base model, tokenizer, hypernet, hc.
      5. Build :class:`OracleAdapterCache` and loop ``num_steps`` records,
         calling :func:`_training_step_round2` per step. Skip sentinels
         advance the step counter without an optimizer step.
      6. Apply the same checkpoint + kill-switch cadence as round-1.

    Args:
        config: :class:`Round2TrainConfig` instance.
        kill_switch_evaluate_fn: Optional zero-arg callable returning Pass@1
            for the kill-switch. When ``None`` and ``config.kill_switch_enabled``
            is True, the default benchmark evaluator is constructed.

    Returns:
        Report dict. Shape depends on mode:
          - dry_run: ``{"dry_run": True, "coverage_ratio", "bin_counts",
            "num_records"}``
          - full: ``{"dry_run": False, "coverage_ratio", "bin_counts",
            "num_records", "final_loss", "best_loss", "steps_completed",
            "kill_switch_triggered"}``
    """
    from model_training.oracle_cache import (  # noqa: PLC0415
        OracleAdapterCache,
        audit_oracle_coverage,
    )

    if not config.dataset_path:
        raise ValueError("config.dataset_path is required for round-2 training")

    records = _load_records(config.dataset_path)
    registry = _open_registry(config.oracle_registry_url)
    coverage_ratio, bin_counts = audit_oracle_coverage(records, registry)

    logger.info(
        "Round-2 oracle coverage: %.3f (min=%.3f); per-bin counts: %s",
        coverage_ratio,
        config.min_oracle_coverage,
        bin_counts,
    )

    if coverage_ratio < config.min_oracle_coverage:
        raise RuntimeError(
            f"Round-2 oracle coverage {coverage_ratio:.3f} < "
            f"min_oracle_coverage {config.min_oracle_coverage:.3f}; aborting. "
            f"Per-bin counts: {bin_counts}"
        )

    if config.dry_run:
        return {
            "dry_run": True,
            "coverage_ratio": coverage_ratio,
            "bin_counts": bin_counts,
            "num_records": len(records),
        }

    handles = _setup_training(config)
    base_model = handles["base_model"]
    tokenizer = handles["tokenizer"]
    hypernet = handles["hypernet"]
    hc = handles["hc"]

    oracle_cache = OracleAdapterCache(
        registry=registry,
        hc=hc,
        max_loaded=config.max_loaded_oracles,
    )

    return _run_training_loop(
        config=config,
        records=records,
        base_model=base_model,
        tokenizer=tokenizer,
        hypernet=hypernet,
        hc=hc,
        oracle_cache=oracle_cache,
        kill_switch_evaluate_fn=kill_switch_evaluate_fn,
        coverage_ratio=coverage_ratio,
        bin_counts=bin_counts,
    )


def _run_training_loop(
    *,
    config: Any,
    records: list[dict[str, Any]],
    base_model: Any,
    tokenizer: Any,
    hypernet: Any,
    hc: Any,
    oracle_cache: Any,
    kill_switch_evaluate_fn: Any | None,
    coverage_ratio: float,
    bin_counts: dict[str, int],
) -> dict[str, Any]:
    """Inner training loop: optimizer, scheduler, checkpoint, kill-switch."""
    import torch  # noqa: PLC0415
    from torch.optim import AdamW  # noqa: PLC0415
    from torch.optim.lr_scheduler import (  # noqa: PLC0415
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    from model_training.d2l_train import (  # noqa: PLC0415
        _save_checkpoint,
        _setup_mlflow,
    )
    from model_training.kill_switch import (  # noqa: PLC0415
        KillSwitchConfig,
        KillSwitchState,
        maybe_run_kill_switch,
    )

    _setup_mlflow(config)

    trainable_params = [p for p in hypernet.parameters() if p.requires_grad]
    logger.info(
        "Round-2 trainable params: %d",
        sum(p.numel() for p in trainable_params),
    )
    optimizer = AdamW(trainable_params, lr=config.lr)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=config.warmup_steps),
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, config.num_steps - config.warmup_steps),
                eta_min=1e-6,
            ),
        ],
        milestones=[config.warmup_steps],
    )

    ks_config = KillSwitchConfig(
        enabled=config.kill_switch_enabled,
        step_cadence=config.kill_switch_step_cadence,
        benchmark_id=config.kill_switch_benchmark_id,
        max_samples=config.kill_switch_max_samples,
        delta=config.kill_switch_delta,
    )
    ks_state = KillSwitchState()

    best_loss = float("inf")
    last_loss = 0.0
    steps_completed = 0
    triggered = False

    for step in range(1, config.num_steps + 1):
        record = records[(step - 1) % len(records)]
        loss, metrics = _training_step_round2(
            record=record,
            base_model=base_model,
            tokenizer=tokenizer,
            hypernet=hypernet,
            hc=hc,
            config=config,
            oracle_cache=oracle_cache,
        )
        if loss is None:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        last_loss = metrics["total_loss"]
        if last_loss < best_loss:
            best_loss = last_loss

        if step % config.checkpoint_every == 0:
            _save_checkpoint(
                step=step,
                hypernet=hypernet,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                hc=hc,
                best_loss=best_loss,
                full=(step % config.full_checkpoint_every == 0),
            )

        steps_completed = step

        if kill_switch_evaluate_fn is not None and maybe_run_kill_switch(
            step=step,
            config=ks_config,
            state=ks_state,
            evaluate_fn=kill_switch_evaluate_fn,
        ):
            logger.error("Round-2 kill-switch triggered at step %d; halting", step)
            triggered = True
            break

    # --- Final checkpoint + registry write-back (only when training ran) ---
    round2_adapter_id: str | None = None
    if steps_completed > 0:
        final_ckpt = _save_checkpoint(
            step=steps_completed,
            hypernet=hypernet,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            hc=hc,
            best_loss=best_loss,
            full=True,
        )
        round2_adapter_id = register_round2_adapter(
            registry=_open_registry(config.oracle_registry_url),
            bin_counts=bin_counts,
            adapter_file_path=str(final_ckpt),
            base_model_id=config.base_model_name,
            rank=config.lora_r,
        )

    return {
        "dry_run": False,
        "coverage_ratio": coverage_ratio,
        "bin_counts": bin_counts,
        "num_records": len(records),
        "final_loss": last_loss,
        "best_loss": best_loss,
        "steps_completed": steps_completed,
        "kill_switch_triggered": triggered,
        "round2_adapter_id": round2_adapter_id,
    }


# ---------------------------------------------------------------------------
# Registry write-back
# ---------------------------------------------------------------------------

ROUND2_GENERATION: int = 2
ROUND2_SOURCE: str = "distillation"


def register_round2_adapter(
    *,
    registry: Any,
    bin_counts: dict[str, int],
    adapter_file_path: str,
    base_model_id: str,
    rank: int,
) -> str:
    """Register the round-2 adapter output in AdapterRegistry.

    The adapter_id is deterministic-per-run: ``round2_<uuid4[:8]>``. Lineage
    is captured as ``parent_ids = json.dumps(sorted(oracle_ids))`` where
    ``oracle_ids`` covers every bin that contributed at least one training
    record. ``generation`` is set to 2 so lineage queries can distinguish
    round-2 output from round-1 oracles (generation 0).

    Args:
        registry: :class:`AdapterRegistry` instance.
        bin_counts: Per-bin record counts from the audit report.
        adapter_file_path: On-disk path to the saved round-2 adapter.
        base_model_id: Base model this adapter was trained against.
        rank: LoRA rank used during training.

    Returns:
        The registered adapter_id.
    """
    from adapter_registry.models import AdapterRecord  # noqa: PLC0415

    adapter_id = f"round2_{uuid4().hex[:8]}"
    parent_ids = sorted(f"oracle_{bk}" for bk in bin_counts)

    file_path = Path(adapter_file_path)
    if file_path.exists() and file_path.is_file():
        file_bytes = file_path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        file_size = len(file_bytes)
    else:
        file_hash = ""
        file_size = 0

    record = AdapterRecord(
        id=adapter_id,
        version=1,
        task_type="round2_hypernet",
        base_model_id=base_model_id,
        rank=rank,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        file_path=str(file_path),
        file_hash=file_hash,
        file_size_bytes=file_size,
        source=ROUND2_SOURCE,
        session_id="",
        parent_ids=json.dumps(parent_ids),
        generation=ROUND2_GENERATION,
    )
    registry.store(record)
    logger.info(
        "Registered round-2 adapter %s with %d parent oracles",
        adapter_id,
        len(parent_ids),
    )
    return adapter_id
