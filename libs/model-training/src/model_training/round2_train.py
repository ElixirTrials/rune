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
    """Thin wrapper so tests can monkeypatch functional LoRA injection."""
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
