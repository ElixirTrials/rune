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
    # Early-stop guardrails (reviewer): abort a long run that is learning nothing
    # useful or damaging the preservation region.
    early_stop_warmup: int = 150
    min_diff_agreement: float = 0.02
    # Smoke #3 showed steady-state preservation ~0.9; 0.7 is a meaningful collapse
    # floor that won't false-abort on normal per-row fluctuation (reviewer).
    min_preservation: float = 0.7
    max_skip_frac: float = 0.5
    # Memory: a frozen 9B bf16 base (~18GB) + trainable hypernet + optimizer +
    # activations does not fit 22GB. 4-bit NF4 base (QLoRA) frees ~13GB and is the
    # primary lever; 8-bit Adam shrinks optimizer state ~4x. Gradient checkpointing
    # is INCOMPATIBLE with the monkeypatched functional-LoRA forward (checkpoint
    # tensor-count mismatch on recompute), so it defaults off.
    load_in_4bit: bool = True
    gradient_checkpointing: bool = False
    use_8bit_optim: bool = True
    # Optimization regime (loss-investigation fixes): the corpus is repo-grouped, so
    # shuffle to decorrelate consecutive batch-1 steps; accumulate gradients to
    # average over per-row difficulty variance; skip zero-diff rows (no signal);
    # evaluate on a held-out split for the honest, ordering-independent progress curve.
    shuffle: bool = True
    seed: int = 0
    grad_accum_steps: int = 8
    weight_decay: float = 0.01  # AdamW regularizer (doc2lora default); anti-overfit
    skip_zero_diff: bool = True
    val_corpus_path: str = ""
    val_sample: int = 40
    val_steps: int = 200


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
        diff_token_fraction,
        preservation_agreement,
        should_early_stop,
        summarize_named_tensors,
    )

    cfg = config if isinstance(config, DistillConfig) else DistillConfig(**dict(config))

    mlflow = _try_mlflow(cfg)

    free = subprocess.run(["free", "-g"], capture_output=True, text=True, check=False)
    logger.info("free -g:\n%s", free.stdout)

    # 1. Base model (frozen) + tokenizer. 4-bit NF4 (QLoRA) by default for memory.
    base_model: Any
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        quant = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": cfg.device},
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        base_model = base_model.to(cfg.device)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.config.use_cache = False
    _can_ckpt = hasattr(base_model, "gradient_checkpointing_enable")
    if cfg.gradient_checkpointing and _can_ckpt:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # HF only checkpoints when module.training is True; params stay frozen.
        base_model.train()
        # Reviewer: train() is memory-only ONLY if no stochastic path. Verify the
        # base is deterministic; otherwise teacher/base logits are unreliable.
        if not _base_is_deterministic(base_model, cfg.device):
            logger.warning(
                "base non-deterministic in train() (dropout?); disabling "
                "gradient checkpointing and using eval()"
            )
            base_model.gradient_checkpointing_disable()
            base_model.eval()
    else:
        base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    # 2. Hypernetwork, re-initialized out of the collapse basin (#49 §A).
    hypernet = load_hypernetwork(
        HypernetworkConfig(checkpoint_path=cfg.checkpoint_path), device=cfg.device
    )
    reinit_scaler_b_nonzero(hypernet, cfg.scaler_b_init)
    hypernet.train()

    layer_indices = list(hypernet.config.layer_indices)

    # 3. Optimizer over trainable hypernet params; assert scaler_B is covered.
    trainable = [p for p in hypernet.parameters() if p.requires_grad]
    optimizer = _build_optimizer(trainable, cfg)
    # Watch scaler_B + scaler_A + a head param so their value trajectories (not just
    # gradients) are logged — reveals whether 8-bit Adam actually updates the
    # collapse-critical params (reviewer).
    watched: dict[str, Any] = {}
    if hasattr(hypernet, "scaler_B"):
        watched["scaler_B"] = hypernet.scaler_B[next(iter(hypernet.scaler_B.keys()))]
    if hasattr(hypernet, "scaler_A"):
        watched["scaler_A"] = hypernet.scaler_A[next(iter(hypernet.scaler_A.keys()))]
    for name, p in hypernet.named_parameters():
        if "head" in name and p.requires_grad:
            watched["head"] = p
            break
    assert_optimizer_covers(watched, optimizer)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "distill_metrics.jsonl"

    _corpus_stats(cfg.corpus_path)
    records = [m for rec in _iter_corpus(cfg.corpus_path) if (m := _map_record(rec))]
    logger.info("loaded %d mapped training records", len(records))
    val_records: list[dict[str, str]] = []
    if cfg.val_corpus_path:
        val_records = [
            m for rec in _iter_corpus(cfg.val_corpus_path) if (m := _map_record(rec))
        ]
        logger.info("loaded %d held-out val records", len(val_records))

    step = 0          # optimizer steps
    micro = 0         # records consumed (micro-steps)
    skipped = 0
    recent_da: list[float] = []
    recent_pres: list[float] = []
    best_val = -1.0   # best held-out val_diff_agreement -> checkpoint_best.pt
    stop_reason: str | None = None
    optimizer.zero_grad()
    accum = 0
    with log_path.open("w") as logf:
        for epoch in range(cfg.num_epochs):
            if stop_reason is not None:
                break
            epoch_records = (
                _shuffled(records, cfg.seed, epoch) if cfg.shuffle else records
            )
            for mapped in epoch_records:
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    break
                context, answer = mapped["context"], mapped["answer"]
                micro += 1
                # Reset per micro-step so a logged gpu_peak_gb reflects THIS row's
                # peak (not a window high-water mark) -> corr(peak, ans_len) is
                # meaningful (reviewer).
                torch.cuda.reset_peak_memory_stats()

                teacher_logits, base_logits, ans_ids = _teacher_base_logits(
                    base_model, tokenizer, context, answer, cfg.max_seq_length
                )
                teacher_top1 = teacher_logits.argmax(dim=-1)
                base_top1 = base_logits.argmax(dim=-1)
                # Skip rows with no diff positions: the masked objective has no signal
                # there (loss==0), so they only add batch-1 noise / wasted compute.
                if cfg.skip_zero_diff and int((base_top1 != teacher_top1).sum()) == 0:
                    skipped += 1
                    del teacher_logits, base_logits
                    continue
                labels = torch.ones_like(teacher_top1)

                lora_dict = _generate_lora_dict(
                    hypernet, context, base_model, tokenizer,
                    layer_indices, cfg.max_seq_length,
                )
                student_logits = _student_logits(
                    base_model, tokenizer, ans_ids, lora_dict,
                    layer_indices, cfg.train_scaling,
                )

                loss = distill_step_loss(
                    student_logits, teacher_logits, base_top1, teacher_top1,
                    labels, k=cfg.topk,
                )
                if cfg.l1_reg_coef > 0.0:
                    l1 = sum(
                        w["A"].abs().sum() + w["B"].abs().sum()
                        for w in lora_dict.values()
                    )
                    loss = loss + cfg.l1_reg_coef * l1
                if not loss.requires_grad:
                    skipped += 1
                    continue

                # Gradient accumulation: average over grad_accum_steps rows to damp
                # the per-row difficulty variance before each optimizer update.
                (loss / cfg.grad_accum_steps).backward()
                accum += 1
                did_step = accum >= cfg.grad_accum_steps
                grad_norms: dict[str, float] = {}
                if did_step:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                    # Capture grad norms BEFORE zero_grad, else they read as empty
                    # (the grad-accum logging bug): confirms scaler_B/head get gradient.
                    grad_norms = _grad_norm_summary(hypernet)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0
                    step += 1

                if did_step and step % cfg.log_steps == 0:
                    student_top1 = student_logits.argmax(dim=-1)
                    da = diff_agreement(student_top1, teacher_top1, base_top1)
                    pres = preservation_agreement(student_top1, teacher_top1, base_top1)
                    recent_da.append(da)
                    recent_pres.append(pres)
                    recent_da[:] = recent_da[-20:]
                    recent_pres[:] = recent_pres[-20:]
                    rec = {
                        "step": step,
                        "micro": micro,
                        "loss": float(loss.detach()),
                        "diff_agreement": da,
                        "preservation": pres,
                        "diff_token_frac": diff_token_fraction(base_top1, teacher_top1),
                        "skipped": skipped,
                        "ans_len": float(len(ans_ids)),
                        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                        **summarize_named_tensors(watched),
                        **grad_norms,
                    }
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    if mlflow is not None:
                        mlflow.log_metrics(
                            {k: v for k, v in rec.items()
                             if isinstance(v, (int, float))},
                            step=step,
                        )
                    logger.info(
                        "step=%d loss=%.4f diff_agr=%.3f pres=%.3f difftok=%.3f",
                        step, rec["loss"], da, pres, rec["diff_token_frac"],
                    )
                    stop_reason = should_early_stop(
                        step, cfg.early_stop_warmup, recent_da, recent_pres,
                        skipped, skipped + micro,
                        min_diff_agreement=cfg.min_diff_agreement,
                        min_preservation=cfg.min_preservation,
                        max_skip_frac=cfg.max_skip_frac,
                    )
                    if stop_reason is not None:
                        logger.warning("EARLY STOP at step %d: %s", step, stop_reason)
                        if mlflow is not None:
                            mlflow.set_tag("early_stop_reason", stop_reason)
                        break

                # Periodic held-out val eval — the ordering/difficulty-independent
                # progress curve (train-loss is confounded by per-row difficulty).
                if did_step and val_records and step % cfg.val_steps == 0:
                    vmet = _eval_on_split(
                        base_model, hypernet, tokenizer, val_records,
                        layer_indices, cfg, hypernet_topk=cfg.topk,
                    )
                    if mlflow is not None:
                        mlflow.log_metrics(vmet, step=step)
                    logger.info("VAL step=%d %s", step, json.dumps(vmet))
                    # Keep the BEST-val checkpoint (anti-overfit: the final-step
                    # checkpoint may be past the val peak).
                    if vmet["val_diff_agreement"] > best_val:
                        best_val = vmet["val_diff_agreement"]
                        _save_checkpoint(hypernet, cfg, step, ckpt_dir,
                                         name="checkpoint_best.pt")
                        if mlflow is not None:
                            mlflow.set_tag("best_val_diff_agreement", f"{best_val:.4f}")
                            mlflow.set_tag("best_val_step", str(step))

                del lora_dict, teacher_logits, base_logits, student_logits
                gc.collect()
                torch.cuda.empty_cache()

                if did_step and cfg.save_steps and step % cfg.save_steps == 0:
                    _save_checkpoint(hypernet, cfg, step, ckpt_dir)

    _save_checkpoint(hypernet, cfg, step, ckpt_dir)
    logger.info("distillation complete: steps=%d skipped=%d", step, skipped)
    if mlflow is not None:
        mlflow.set_tag("final_step", str(step))
        mlflow.end_run()


def _base_is_deterministic(base_model: Any, device: str) -> bool:
    """Two identical forwards must match — guards train()-mode dropout (reviewer)."""
    import torch  # noqa: PLC0415

    ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    with torch.no_grad():
        a = base_model(ids, use_cache=False).logits
        b = base_model(ids, use_cache=False).logits
    return bool(torch.allclose(a, b, atol=1e-4))


def _shuffled(records: list[Any], seed: int, epoch: int) -> list[Any]:
    """Deterministic per-epoch shuffle (decorrelate repo-grouped consecutive rows)."""
    import random  # noqa: PLC0415

    out = list(records)
    random.Random(seed + epoch).shuffle(out)
    return out


def _eval_on_split(
    base_model: Any,
    hypernet: Any,
    tokenizer: Any,
    val_records: list[dict[str, str]],
    layer_indices: list[int],
    cfg: DistillConfig,
    hypernet_topk: int,
) -> dict[str, float]:
    """Held-out diff_agreement/preservation on val families (never trained on).

    The honest, ordering/difficulty-independent progress curve — train-loss is
    confounded by per-row diff-token count (corr ~0.88). Eval-only (no grad).
    """
    import statistics  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from rune.training.collapse_metrics import (  # noqa: PLC0415
        diff_agreement,
        preservation_agreement,
    )

    hypernet.eval()
    das: list[float] = []
    press: list[float] = []
    with torch.no_grad():
        for m in val_records[: cfg.val_sample]:
            ctx, ans = m["context"], m["answer"]
            t, b, ans_ids = _teacher_base_logits(
                base_model, tokenizer, ctx, ans, cfg.max_seq_length
            )
            tt, bt = t.argmax(dim=-1), b.argmax(dim=-1)
            if int((bt != tt).sum()) == 0:
                continue
            ld = _generate_lora_dict(
                hypernet, ctx, base_model, tokenizer, layer_indices, cfg.max_seq_length
            )
            s = _student_logits(
                base_model, tokenizer, ans_ids, ld, layer_indices, cfg.train_scaling
            )
            stt = s.argmax(dim=-1)
            das.append(diff_agreement(stt, tt, bt))
            press.append(preservation_agreement(stt, tt, bt))
            del ld, t, b, s
    hypernet.train()
    return {
        "val_diff_agreement": statistics.mean(das) if das else 0.0,
        "val_preservation": statistics.mean(press) if press else 1.0,
        "val_n": float(len(das)),
    }


def _build_optimizer(params: list[Any], cfg: DistillConfig) -> Any:
    """8-bit Adam (bitsandbytes) when enabled+available, else AdamW.

    8-bit Adam shrinks optimizer state ~4x — material headroom when a frozen 9B
    base already fills most of a 22GB GPU.
    """
    import torch  # noqa: PLC0415

    if cfg.use_8bit_optim:
        try:
            import bitsandbytes as bnb  # noqa: PLC0415

            logger.info("optimizer: 8-bit Adam (bitsandbytes), wd=%s", cfg.weight_decay)
            opt8 = bnb.optim.Adam8bit  # type: ignore[attr-defined]
            return opt8(params, lr=cfg.learning_rate,  # type: ignore[no-untyped-call]
                        weight_decay=cfg.weight_decay)
        except Exception as exc:  # noqa: BLE001
            logger.warning("8-bit Adam unavailable (%s); falling back to AdamW", exc)
    return torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )


def _try_mlflow(cfg: Any) -> Any:
    """Start an MLflow run for live training monitoring; None if unavailable.

    Tracking URI defaults to the repo's live server (localhost:5000) but honours
    MLFLOW_TRACKING_URI. Never fatal — a monitoring backend must not break training.
    """
    import os  # noqa: PLC0415

    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(cfg.experiment_name)
        mlflow.start_run(run_name=cfg.experiment_name)
        keys = ("model_id", "corpus_path", "learning_rate", "num_epochs",
                "max_seq_length", "scaler_b_init", "train_scaling", "topk",
                "l1_reg_coef", "max_steps", "early_stop_warmup",
                "min_diff_agreement", "min_preservation", "max_skip_frac")
        mlflow.log_params({k: getattr(cfg, k, None) for k in keys})
        logger.info("MLflow run started (experiment=%s)", cfg.experiment_name)
        return mlflow
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow disabled (%s); continuing with JSONL only", exc)
        return None


def _iter_corpus(path: str) -> Any:
    """Yield raw JSON records from a JSONL corpus."""
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _map_record(rec: dict[str, Any]) -> dict[str, str] | None:
    """Map a raw corpus record to {context, answer}, or None if unusable.

    Two schemas are supported:
      - synthetic: {"context", "answer"} used directly.
      - S3 github-pairs: {"activation_text", "teacher_text", ...}. teacher_text
        is activation_text + the "## Revision ..." block, so context =
        activation_text and answer = teacher_text minus the activation_text prefix
        (the revision the adapter must reproduce). If teacher_text does not start
        with activation_text (unexpected), fall back to the whole teacher_text.
    Returns None when context or answer is empty after stripping.
    """
    if "context" in rec and "answer" in rec:
        ctx, ans = str(rec.get("context", "")), str(rec.get("answer", ""))
    else:
        at = rec.get("activation_text")
        tt = rec.get("teacher_text")
        if at is None or tt is None:
            return None
        at, tt = str(at), str(tt)
        ctx = at
        ans = tt[len(at):] if tt.startswith(at) else tt
    ctx, ans = ctx.strip(), ans.strip()
    if not ctx or not ans:
        return None
    return {"context": ctx, "answer": ans}


def _corpus_stats(path: str) -> dict[str, Any]:
    """One-pass row accounting (reviewer).

    Beyond raw/mapped/skipped, reports the prefix-strip health that the S3 mapper
    depends on: exact-prefix rate (teacher_text starts with activation_text) vs
    fallback rate (it does not — brittle to whitespace/template drift), the answer
    char-length distribution after stripping, and a few sampled fallback task_ids
    so silent template drift can't pass unnoticed. (Token-length / diff-token
    fraction need the tokenizer + base model — those live in the GPU readiness pass.)
    """
    raw = mapped = empty_ctx = empty_ans = 0
    exact_prefix = fallback = s3_rows = 0
    ans_lens: list[int] = []
    fallback_samples: list[str] = []
    for rec in _iter_corpus(path):
        raw += 1
        if "context" in rec and "answer" in rec:
            ctx, ans = str(rec.get("context", "")), str(rec.get("answer", ""))
        else:
            at, tt = rec.get("activation_text"), rec.get("teacher_text")
            if at is None or tt is None:
                continue
            at, tt = str(at), str(tt)
            s3_rows += 1
            if tt.startswith(at):
                exact_prefix += 1
                ctx, ans = at, tt[len(at):]
            else:
                fallback += 1
                if len(fallback_samples) < 5:
                    fallback_samples.append(str(rec.get("task_id", "?")))
                ctx, ans = at, tt
        if not ctx.strip():
            empty_ctx += 1
        if not ans.strip():
            empty_ans += 1
        if _map_record(rec) is not None:
            mapped += 1
            ans_lens.append(len(ans.strip()))
    ans_lens.sort()
    dist = {}
    if ans_lens:
        n = len(ans_lens)
        dist = {"min": ans_lens[0], "median": ans_lens[n // 2],
                "p90": ans_lens[min(n - 1, int(0.9 * n))], "max": ans_lens[-1]}
    stats: dict[str, Any] = {
        "raw": raw, "mapped": mapped, "skipped": raw - mapped,
        "empty_context": empty_ctx, "empty_answer": empty_ans,
        "s3_rows": s3_rows, "exact_prefix": exact_prefix, "fallback": fallback,
        "answer_char_len": dist, "fallback_task_ids": fallback_samples,
    }
    logger.info("corpus stats: %s", stats)
    return stats


def _prepare_ids(
    tokenizer: Any, context: str, answer: str, max_length: int
) -> tuple[list[int], list[int]]:
    """Answer-preserving truncation (issue #49 / reviewer).

    The supervised span is the answer, so it must never be truncated away. Keep the
    FULL answer; spend the remaining budget on the END of the context (the part
    nearest the answer — for code-review rows that is the review feedback / current
    code tail). If the answer alone exceeds max_length, keep the answer HEAD and use
    no context. Returns (full_ids, ans_ids); the answer is always the suffix of
    full_ids so teacher logits over the last len(ans_ids) positions are answer tokens.
    """
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(ans_ids) >= max_length:
        ans_ids = ans_ids[:max_length]
        return list(ans_ids), list(ans_ids)
    budget = max_length - len(ans_ids)
    ctx_ids = ctx_ids[-budget:]
    return list(ctx_ids) + list(ans_ids), list(ans_ids)


def _teacher_base_logits(
    base_model: Any,
    tokenizer: Any,
    context: str,
    answer: str,
    max_length: int,
) -> Any:
    """Teacher (context+answer) and base (answer-only) logits over the answer span.

    Uses answer-preserving truncation so the supervised span is always answer
    tokens (issue #49 / reviewer). Returns (teacher_logits, base_logits, ans_ids);
    both logit tensors cover exactly the len(ans_ids) answer positions and are
    position-aligned with the student's answer-only forward.
    """
    import torch  # noqa: PLC0415

    device = next(base_model.parameters()).device
    full_ids, ans_ids = _prepare_ids(tokenizer, context, answer, max_length)
    ans_len = len(ans_ids)
    full = torch.tensor([full_ids], device=device)
    ans_only = torch.tensor([ans_ids], device=device)

    with torch.no_grad():
        ctx_disable = (
            base_model.disable_adapter()
            if hasattr(base_model, "disable_adapter")
            else _nullctx()
        )
        with ctx_disable:
            teacher = base_model(full, use_cache=False).logits[0, -ans_len:]
            base = base_model(ans_only, use_cache=False).logits[0, -ans_len:]
    return teacher, base, ans_ids


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


def _lora_delta(x: Any, a: Any, b: Any, scaling: float) -> Any:
    """LoRA delta ``(x @ Aᵀ) @ B * scaling`` for a single context (n_ctx=1).

    a: [1, r, d_in], b: [1, r, d_out], x: [1, seq, d_in] -> [1, seq, d_out].
    Pure (testable) — this is the equivalence contract for the custom functional
    forward used with a quantized (bnb Linear4bit) base.
    """
    import torch  # noqa: PLC0415

    xd = x.to(a.dtype)
    delta = torch.einsum("c r i, c s i -> c s r", a, xd)
    return torch.einsum("c r o, c s r -> c s o", b, delta) * scaling


def _functional_lora(
    base_model: Any,
    layer_indices: list[int],
    lora_dict: dict[str, Any],
    scaling: float,
    n_qs: Any,
) -> Any:
    """Context manager applying ``lora_dict`` functionally to the base model.

    Patches each target ``Linear.forward`` to add ``(x @ Aᵀ) @ B * scaling`` on top
    of the layer's ORIGINAL output, using the grad-carrying generated A/B, then
    restores the original forwards on exit.

    The base_out comes from the layer's own ``forward_orig`` (not
    ``torch.nn.Linear.forward``) so this works for ANY base layer type — including a
    4-bit ``bitsandbytes.Linear4bit`` (QLoRA), whose weights the plain nn.Linear
    forward cannot read. Indexing is POSITIONAL — the lora_dict tensor's layer axis
    has length ``len(layer_indices)`` (built positionally), so ``[:, layer_pos]`` is
    correct even for non-contiguous selected layers (the package's
    ``apply_lora_to_layers`` indexes by absolute layer id and would misapply).
    """
    import contextlib  # noqa: PLC0415
    from operator import attrgetter  # noqa: PLC0415

    from ctx_to_lora.utils import get_layers  # noqa: PLC0415

    _ATTN = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}

    def _make_forward(orig: Any, a: Any, b: Any) -> Any:
        def _fwd(x: Any, *args: Any, **kwargs: Any) -> Any:
            base_out = orig(x, *args, **kwargs)
            return base_out + _lora_delta(x, a, b, scaling).to(base_out.dtype)
        return _fwd

    @contextlib.contextmanager
    def _ctx() -> Any:
        layers = get_layers(base_model)
        patched: list[Any] = []
        try:
            for layer_pos, layer_idx in enumerate(layer_indices):
                layer = layers[layer_idx]
                for mname, w in lora_dict.items():
                    long = f"self_attn.{mname}" if mname in _ATTN else f"mlp.{mname}"
                    module = attrgetter(long)(layer)
                    module.forward_orig = module.forward
                    module.forward = _make_forward(
                        module.forward_orig, w["A"][:, layer_pos], w["B"][:, layer_pos]
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
    ans_ids: list[int],
    lora_dict: dict[str, Any],
    layer_indices: list[int],
    scaling: float,
) -> Any:
    """Student logits over the answer span with the generated adapter applied.

    Takes the prepared ans_ids (from _teacher_base_logits) so the student's
    supervised span matches the teacher's exactly. The adapter is applied
    functionally (not via load_state_dict) so the generated A/B tensors stay in the
    autograd graph and gradients flow to the hypernetwork.
    """
    import torch  # noqa: PLC0415

    device = next(base_model.parameters()).device
    ans_only = torch.tensor([ans_ids], device=device)
    n_qs = torch.tensor([1], device=device)
    with _functional_lora(base_model, layer_indices, lora_dict, scaling, n_qs):
        return base_model(ans_only, use_cache=False).logits[0]


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
    hypernet: Any, cfg: DistillConfig, step: int, ckpt_dir: Any,
    name: str = "checkpoint.pt",
) -> None:
    """Persist the hypernetwork state dict + config + step."""
    import torch  # noqa: PLC0415

    path = ckpt_dir / name
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
    # Memory: do NOT materialize a full-vocab fp32 copy of the logits — at vocab
    # 151936 that is ~0.5GB per [N,V] tensor, and for the (grad-carrying) student it
    # is retained in the autograd graph. logsumexp reduces over the vocab without a
    # persistent fp32 copy; we upcast only the small gathered [N,K] slices.
    t_denom = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
    teacher_logp = (topk_vals - t_denom).float()  # [N, K]
    teacher_p = teacher_logp.exp()  # [N, K]
    s_denom = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    student_logq = (student_logits.gather(-1, topk_idx) - s_denom).float()  # [N, K]
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
