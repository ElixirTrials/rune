"""D2L privileged-context self-distillation for the HyperLoRA hypernetwork.

Teacher = frozen base model with the trajectory in-context (adapters disabled);
student = base + generated adapter with the trajectory removed from the prompt.
Loss = top-K KL over the answer span, masked to diff tokens (where teacher != base).

GPU imports are deferred; only pure tensor helpers are import-safe.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from rune.training.d2l_train import D2LTrainConfig

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class DistillConfig(D2LTrainConfig):
    """Configuration for the D2L context-distillation loop (issue #49).

    Inherits corpus/checkpoint/optimizer fields from ``D2LTrainConfig`` and adds
    the D2L-specific knobs. ``l1_reg_coef`` defaults to 0.0 because the L1 sink
    pushed the adapter into the zero-collapse basin (#49 §A); ``scaler_b_init``
    re-initializes the gate away from that basin; ``topk`` bounds the KL support.

    Attributes:
        l1_reg_coef: Coefficient on the L1 norm of generated weights. 0 disables
            the L1 sink (#49 §A).
        scaler_b_init: Constant value used to re-initialize ``scaler_B`` at train
            start (#49 §A).
        topk: Number of teacher top tokens matched by the KL loss.
        max_steps: Optional cap on training steps (None = full corpus * epochs).
        grad_clip: Max gradient norm for clipping.
        log_steps: Cadence (in steps) for diagnostic JSONL logging.
        device: Target device for base model + hypernetwork.
    """

    l1_reg_coef: float = 0.0
    scaler_b_init: float = 1.0
    topk: int = 50
    max_steps: int | None = None
    grad_clip: float = 1.0
    log_steps: int = 10
    device: str = "cuda"
    train_scaling: float = 2.0


def run_hypernet_distillation(config: Any) -> None:
    """Stage-2 entrypoint: D2L privileged-context self-distillation.

    Teacher = frozen base model with the trajectory in-context (adapters
    disabled); student = base + hypernetwork-generated adapter with the
    trajectory removed from the prompt. The loss is top-K KL over the answer
    span, masked to diff tokens (positions where teacher disagrees with base).

    This is a GPU run-and-observe path (gated by the Stage-0 synthetic overfit
    harness, not by CI); all GPU imports are deferred inside the body so the
    module stays CPU-importable.

    Args:
        config: A ``DistillConfig`` (or compatible ``D2LTrainConfig``) instance.
    """
    import gc  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        load_hypernetwork,
        reinit_scaler_b_nonzero,
    )
    from rune.training.collapse_metrics import (  # noqa: PLC0415
        assert_optimizer_covers,
        diff_agreement,
        summarize_named_tensors,
    )

    cfg = config if isinstance(config, DistillConfig) else DistillConfig(**dict(config))

    free = subprocess.run(["free", "-g"], capture_output=True, text=True, check=False)
    logger.info("free -g:\n%s", free.stdout)

    # 1. Base model (frozen) + tokenizer.
    base_model: Any = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    base_model = base_model.to(cfg.device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    # 2. Hypernetwork, re-initialized out of the collapse basin (#49 §A).
    hypernet = load_hypernetwork(
        HypernetworkConfig(checkpoint_path=cfg.checkpoint_path), device=cfg.device
    )
    reinit_scaler_b_nonzero(hypernet, cfg.scaler_b_init)
    hypernet.train()

    layer_indices = list(hypernet.config.lora_config.layers)

    # 3. Optimizer over trainable hypernet params; assert scaler_B is covered.
    trainable = [p for p in hypernet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    watched: dict[str, Any] = {}
    if hasattr(hypernet, "scaler_B"):
        first = next(iter(hypernet.scaler_B.keys()))
        watched["scaler_B"] = hypernet.scaler_B[first]
    assert_optimizer_covers(watched, optimizer)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "distill_metrics.jsonl"

    step = 0
    skipped = 0
    with log_path.open("w") as logf:
        for _epoch in range(cfg.num_epochs):
            for record in _iter_corpus(cfg.corpus_path):
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    break
                context = record["context"]
                answer = record["answer"]

                # Teacher: base + context + answer, adapters disabled.
                teacher_logits, base_logits, ans_slice = _teacher_base_logits(
                    base_model, tokenizer, context, answer, cfg.max_seq_length
                )
                teacher_top1 = teacher_logits.argmax(dim=-1)
                base_top1 = base_logits.argmax(dim=-1)
                labels = torch.ones_like(teacher_top1)

                # Student: base + generated adapter (grad-carrying), answer-only
                # prompt. lora_dict comes straight from generate_weights (NOT the
                # PEFT export) and is applied functionally so autograd flows back
                # to the hypernetwork.
                lora_dict = _generate_lora_dict(
                    hypernet, context, base_model, tokenizer,
                    layer_indices, cfg.max_seq_length,
                )
                student_logits = _student_logits(
                    base_model, tokenizer, answer, lora_dict,
                    ans_slice, layer_indices, cfg.train_scaling,
                )

                loss = distill_step_loss(
                    student_logits,
                    teacher_logits,
                    base_top1,
                    teacher_top1,
                    labels,
                    k=cfg.topk,
                )
                if cfg.l1_reg_coef > 0.0:
                    l1 = sum(
                        w["A"].abs().sum() + w["B"].abs().sum()
                        for w in lora_dict.values()
                    )
                    loss = loss + cfg.l1_reg_coef * l1

                if not loss.requires_grad:
                    skipped += 1
                    step += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step()

                if step % cfg.log_steps == 0:
                    student_top1 = student_logits.argmax(dim=-1)
                    rec = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "diff_agreement": diff_agreement(
                            student_top1, teacher_top1, base_top1
                        ),
                        "skipped": skipped,
                        **summarize_named_tensors(watched),
                        **_grad_norm_summary(hypernet),
                    }
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    logger.info("step=%d loss=%.4f", step, rec["loss"])

                step += 1
                del lora_dict, teacher_logits, base_logits, student_logits
                gc.collect()
                torch.cuda.empty_cache()

                if cfg.save_steps and step % cfg.save_steps == 0:
                    _save_checkpoint(hypernet, cfg, step, ckpt_dir)

    _save_checkpoint(hypernet, cfg, step, ckpt_dir)
    logger.info("distillation complete: steps=%d skipped=%d", step, skipped)


def _iter_corpus(path: str) -> Any:
    """Yield {context, answer} records from a JSONL corpus."""
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _teacher_base_logits(
    base_model: Any,
    tokenizer: Any,
    context: str,
    answer: str,
    max_length: int,
) -> Any:
    """Teacher (context+answer) and base (answer-only) logits over the answer span."""
    import torch  # noqa: PLC0415

    device = next(base_model.parameters()).device
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    full = torch.tensor([ctx_ids + ans_ids], device=device)[:, :max_length]
    ans_only = torch.tensor([ans_ids], device=device)[:, :max_length]
    ans_len = min(len(ans_ids), max_length)
    ans_slice = slice(-ans_len, None)

    with torch.no_grad():
        ctx_disable = (
            base_model.disable_adapter()
            if hasattr(base_model, "disable_adapter")
            else _nullctx()
        )
        with ctx_disable:
            teacher = base_model(full, use_cache=False).logits[0, ans_slice]
            base = base_model(ans_only, use_cache=False).logits[0, ans_slice]
    return teacher, base, ans_slice


def _generate_lora_dict(
    hypernet: Any,
    context: str,
    base_model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    max_length: int,
) -> Any:
    """Generate the grad-carrying ``{module: {A, B}}`` lora_dict for one context.

    Extracts activations under ``disable_adapter`` (so conditioning is never
    contaminated by a previously-applied adapter), then runs the perceiver WITHOUT
    ``no_grad`` so gradients flow back to the hypernetwork. This is the training
    counterpart of ``generate_adapter_weights`` (which is no_grad + PEFT-keyed and
    is only for inference/hotswap).
    """
    from rune.model.hypernetwork import (  # noqa: PLC0415
        extract_activations_with_model,
    )

    features, attn_mask = extract_activations_with_model(
        text=context,
        model=base_model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        max_length=max_length,
    )
    lora_dict, _ = hypernet.generate_weights(features, attn_mask, None)
    return lora_dict


def _functional_lora(
    base_model: Any,
    layer_indices: list[int],
    lora_dict: dict[str, Any],
    scaling: float,
    n_qs: Any,
) -> Any:
    """Context manager applying ``lora_dict`` functionally to the base model.

    Patches each target ``Linear.forward`` with ctx_to_lora's ``lora_forward``
    (which adds ``B @ A @ x * scaling`` using the grad-carrying tensors), then
    restores the original forwards on exit. Indexing is POSITIONAL — the lora_dict
    tensor's layer axis has length ``len(layer_indices)`` (built positionally by
    ``_to_lora_dict``), so ``[:, layer_pos]`` is correct even when the selected
    layers are non-contiguous (the package's ``apply_lora_to_layers`` indexes by
    absolute layer id and would misapply for non-contiguous layers).
    """
    import contextlib  # noqa: PLC0415
    from functools import partial  # noqa: PLC0415
    from operator import attrgetter  # noqa: PLC0415

    from ctx_to_lora.modeling.lora_layer import lora_forward  # noqa: PLC0415
    from ctx_to_lora.utils import get_layers  # noqa: PLC0415

    _ATTN = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}

    @contextlib.contextmanager
    def _ctx() -> Any:
        layers = get_layers(base_model)
        tot_q = int(n_qs.sum())
        patched: list[Any] = []
        try:
            for layer_pos, layer_idx in enumerate(layer_indices):
                layer = layers[layer_idx]
                for mname, w in lora_dict.items():
                    long = f"self_attn.{mname}" if mname in _ATTN else f"mlp.{mname}"
                    module = attrgetter(long)(layer)
                    module.forward_orig = module.forward
                    module.forward = partial(
                        lora_forward,
                        n_qs=n_qs,
                        tot_q=tot_q,
                        A=w["A"][:, layer_pos],
                        B=w["B"][:, layer_pos],
                        lora_dropout_p=0.0,
                        scaling=scaling,
                        self=module,
                    )
                    patched.append(module)
            yield
        finally:
            for module in patched:
                module.forward = module.forward_orig
                del module.forward_orig

    return _ctx()


def _student_logits(
    base_model: Any,
    tokenizer: Any,
    answer: str,
    lora_dict: dict[str, Any],
    ans_slice: slice,
    layer_indices: list[int],
    scaling: float,
) -> Any:
    """Student logits over the answer span with the generated adapter applied.

    The adapter is applied functionally (not via load_state_dict) so the generated
    A/B tensors stay in the autograd graph and gradients flow to the hypernetwork.
    """
    import torch  # noqa: PLC0415

    device = next(base_model.parameters()).device
    ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ans_only = torch.tensor([ans_ids], device=device)
    n_qs = torch.tensor([1], device=device)
    with _functional_lora(base_model, layer_indices, lora_dict, scaling, n_qs):
        return base_model(ans_only, use_cache=False).logits[0, ans_slice]


def _grad_norm_summary(hypernet: Any) -> dict[str, float]:
    """Per-component (scaler/bias/head) gradient L2 norms for collapse tripwires."""
    out: dict[str, float] = {}
    for name, p in hypernet.named_parameters():
        if p.grad is None:
            continue
        for group in ("scaler_A", "scaler_B", "bias_A", "bias_B", "head"):
            if group in name:
                key = f"{group}/grad_l2"
                out[key] = out.get(key, 0.0) + float(p.grad.norm())
    return out


def _save_checkpoint(
    hypernet: Any, cfg: DistillConfig, step: int, ckpt_dir: Any
) -> None:
    """Persist the hypernetwork state dict + config + step."""
    import torch  # noqa: PLC0415

    path = ckpt_dir / "checkpoint.pt"
    torch.save(
        {
            "hypernet_state_dict": hypernet.state_dict(),
            "hypernet_config": hypernet.config,
            "step": step,
        },
        path,
    )
    logger.info("saved checkpoint step=%d → %s", step, path)


def _nullctx() -> Any:
    """Return a no-op context manager (deferred-import friendly)."""
    import contextlib  # noqa: PLC0415

    return contextlib.nullcontext()


def compute_diff_positions(base_top1: Any, teacher_top1: Any, labels: Any) -> Any:
    """Boolean mask: supervised positions where base and teacher top-1 disagree."""
    return (labels != IGNORE_INDEX) & (base_top1 != teacher_top1)


def topk_kl_loss(student_logits: Any, teacher_logits: Any, k: int = 50) -> Any:
    """KL(teacher || student) over the teacher's top-K tokens, mean over rows.

    Args:
        student_logits: [N, V] student logits at supervised positions.
        teacher_logits: [N, V] teacher logits at the same positions.
        k: number of top teacher tokens to match.
    """
    import torch  # noqa: PLC0415

    k = min(k, teacher_logits.shape[-1])
    topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
    t_denom = torch.logsumexp(teacher_logits.float(), dim=-1, keepdim=True)
    teacher_logp = topk_vals.float() - t_denom  # [N, K]
    teacher_p = teacher_logp.exp()  # [N, K]
    s_denom = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
    student_logq = student_logits.float().gather(-1, topk_idx) - s_denom  # [N, K]
    return (teacher_p * (teacher_logp - student_logq)).sum(dim=-1).mean()


def distill_step_loss(
    student_logits: Any,
    teacher_logits: Any,
    base_top1: Any,
    teacher_top1: Any,
    labels: Any,
    k: int = 50,
) -> Any:
    """Top-K KL restricted to diff positions (base != teacher on supervised tokens).

    Returns a scalar loss. If there are no diff positions, returns 0 (no signal).
    """
    mask = compute_diff_positions(base_top1, teacher_top1, labels)
    if int(mask.sum()) == 0:
        return student_logits.sum() * 0.0
    return topk_kl_loss(student_logits[mask], teacher_logits[mask], k=k)
