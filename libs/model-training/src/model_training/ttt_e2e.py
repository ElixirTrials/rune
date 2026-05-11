"""Test-Time Training (TTT-E2E) baseline for Condition (iv).

Implements the TTT-E2E approach: at inference time, fine-tune a fraction
of MLP layers on the input context before generating the output. This is
the "learn at test time" baseline from Sun et al. 2024.

GPU imports deferred per INFRA-05.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTTConfig:
    """Configuration for TTT-E2E inference-time training.

    Attributes:
        mlp_fraction: Fraction of MLP layers to train (0.25 = 25%).
        inner_lr: Learning rate for the inner (test-time) optimization.
        inner_steps: Number of gradient steps at test time per input.
        max_context_tokens: Maximum context length for TTT input.
    """

    mlp_fraction: float = 0.25
    inner_lr: float = 1e-4
    inner_steps: int = 5
    max_context_tokens: int = 2048


def select_mlp_layers(
    layer_names: list[str],
    fraction: float = 0.25,
) -> list[str]:
    """Select a fraction of MLP layers uniformly spaced across depth.

    Args:
        layer_names: All MLP layer names in the model.
        fraction: Fraction to select (0.25 = every 4th layer).

    Returns:
        Selected layer names.
    """
    fraction = max(0.0, min(1.0, fraction))
    n_select = max(0, round(len(layer_names) * fraction))
    if n_select == 0:
        return []
    if n_select >= len(layer_names):
        return layer_names[:]

    step = len(layer_names) / n_select
    indices = [round(i * step) for i in range(n_select)]
    indices = [min(i, len(layer_names) - 1) for i in indices]
    return [layer_names[i] for i in sorted(set(indices))]


def ttt_forward_pass(
    model: Any,
    tokenizer: Any,
    context: str,
    query: str,
    config: TTTConfig,
) -> dict[str, Any]:
    """Run TTT-E2E: inner-loop train on context, then generate for query.

    Args:
        model: HuggingFace causal LM (must support .parameters()).
        tokenizer: Corresponding tokenizer.
        context: Training context (trajectory/history).
        query: The prompt to complete after TTT.
        config: TTT configuration.

    Returns:
        Dict with "generation", "latency_ms", "inner_loss_final".
    """
    import torch

    all_mlp_names = [
        name
        for name, _ in model.named_parameters()
        if "mlp" in name and "weight" in name
    ]
    selected = select_mlp_layers(all_mlp_names, config.mlp_fraction)

    for name, param in model.named_parameters():
        param.requires_grad = name in selected

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if n in selected],
        lr=config.inner_lr,
    )

    ctx_ids = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_context_tokens,
    ).to(model.device)

    start = time.perf_counter()
    model.train()
    final_loss = 0.0
    for _ in range(config.inner_steps):
        outputs = model(**ctx_ids, labels=ctx_ids["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = loss.item()
    train_time = (time.perf_counter() - start) * 1000

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    query_ids = tokenizer(query, return_tensors="pt").to(model.device)
    gen_start = time.perf_counter()
    with torch.no_grad():
        gen_ids = model.generate(**query_ids, max_new_tokens=512, do_sample=False)
    gen_time = (time.perf_counter() - gen_start) * 1000

    generation = tokenizer.decode(
        gen_ids[0][query_ids["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return {
        "generation": generation,
        "latency_ms": train_time + gen_time,
        "train_latency_ms": train_time,
        "gen_latency_ms": gen_time,
        "inner_loss_final": final_loss,
    }
