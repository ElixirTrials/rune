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

import logging
from typing import Any

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
