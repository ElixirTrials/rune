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
    # Precision: the current 4B base in bf16 (~8GB) + trainable hypernet + optimizer +
    # activations fits the 23GB GPU and MATCHES the engine (wrapper.py loads bf16), so
    # bf16 is the default. 4-bit NF4 (QLoRA) is opt-in for a larger base / tighter
    # memory; 8-bit Adam shrinks optimizer state ~4x. Gradient checkpointing is
    # INCOMPATIBLE with the monkeypatched functional-LoRA forward (checkpoint
    # tensor-count mismatch on recompute), so it defaults off.
    load_in_4bit: bool = False
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
    # Contrastive specificity term (issue #49): force the matched adapter to beat a
    # hard-negative (same row, feedback content swapped) on edit-local gold tokens,
    # so the objective rewards trajectory MEMORY, not a generic edit-booster.
    contrastive: bool = False
    # Contrastive mode:
    # - "feedback_swap_edit_local": issue #49 reviewer objective
    #   (edit-local mask, feedback swapped).
    # - "body_derangement": issue #52 cross-over control
    #   (BODY span, adapter derangement negative).
    # - "body_recall_guarded": issue #52 pilot-2 objective. PRIMARY raises matched
    #   body lp toward matched_target_lp (accessibility); GUARD holds the deranged
    #   partner's body lp at its frozen warm-start baseline (anti generic-boost),
    #   with NO suppression reward. The win can only come from matched rising with
    #   mismatch held — see docs/issue52-crossover-frozen-probe-results-2026-06-03.md.
    #   REMOVE BEFORE MERGE unless the pilot validates the lever — the body_* branches
    #   + helpers (_body_span_mask, _deranged_partner_context,
    #   _contrastive_logprob_readout, _precompute_recall_baselines,
    #   _recall_snapshot) are issue-52 probe code under eval (handoff RBM manifest).
    contrastive_mode: str = "feedback_swap_edit_local"
    contrastive_weight: float = 1.0
    contrastive_margin: float = 1.0
    # body_recall_guarded knobs (issue #52 pilot 2).
    matched_target_lp: float = -0.7  # accessibility floor (warm ~-1.0, oracle -0.22)
    primary_weight: float = 1.0  # lambda_p on relu(target - lp_matched)
    guard_weight: float = 1.0  # lambda_g on relu(lp_mismatch - lp_n0)
    # Frozen-probe + generation snapshot cadence (0 disables). Mirrors absent/present
    # body metric the go/no-go is decided on, plus a valid-code generation canary.
    snapshot_steps: int = 0
    snapshot_episodes: int = 8


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

    from rune.model.adapter_contract import (  # noqa: PLC0415
        assemble_adapter,
        effective_scaling,
    )
    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        load_hypernetwork,
        reinit_scaler_b_nonzero,
        scaler_b_is_collapsed,
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

    # 1. Base model (frozen) + tokenizer. bf16 by default (engine parity); 4-bit NF4
    #    (QLoRA) opt-in via load_in_4bit for a larger base / tighter memory.
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

    # 2. Hypernetwork. ctx_to_lora zero-inits scaler_B (the collapse basin: B =
    # B_raw * scaler_B = 0, no gradient to B_raw), so a from-scratch run must
    # re-init it non-zero. A WARM-START from a trained checkpoint already carries
    # a learned, structured scaler_B (~0.057) that MUST be preserved — clobbering
    # it with 1.0 inflates the B-side ~17x at effective_scaling=lora_alpha and
    # destroys the adapter (60-step smoke: matched-zero -8.8). Re-init ONLY when
    # actually collapsed (#49 §A).
    hypernet = load_hypernetwork(
        HypernetworkConfig(checkpoint_path=cfg.checkpoint_path), device=cfg.device
    )
    if scaler_b_is_collapsed(hypernet):
        reinit_scaler_b_nonzero(hypernet, cfg.scaler_b_init)
        logger.info(
            "scaler_B in collapse basin at load → re-init to %.3f", cfg.scaler_b_init
        )
    elif hasattr(hypernet, "scaler_B"):
        sb0 = hypernet.scaler_B[next(iter(hypernet.scaler_B.keys()))]
        logger.info(
            "scaler_B preserved from warm-start checkpoint (mean|·|=%.4f)",
            float(sb0.detach().abs().mean()),
        )
    hypernet.train()

    layer_indices = list(hypernet.config.layer_indices)
    # Shared apply contract (Sakana parity): effective scaling == lora_alpha (NOT
    # alpha/r and NOT cfg.train_scaling), applied to the combine_lora-assembled
    # adapter. n_chunks is the single-context int tensor combine_lora splits on.
    eff_scaling = effective_scaling(hypernet)
    n_chunks = torch.ones(1, dtype=torch.int32, device=cfg.device)
    logger.info(
        "adapter contract: effective_scaling(lora_alpha)=%.4f use_bias=%s",
        eff_scaling,
        getattr(hypernet.config, "use_bias", False),
    )

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

    from rune.training.contrastive import (  # noqa: PLC0415
        edit_local_mask,
        extract_review_feedback,
        make_hard_negative,
    )

    _corpus_stats(cfg.corpus_path)
    records = []
    for rec in _iter_corpus(cfg.corpus_path):
        m = _map_record(rec)
        if not m:
            continue
        if cfg.contrastive and cfg.contrastive_mode == "feedback_swap_edit_local":
            m["pre_code"] = str(rec.get("pre_code", ""))
            m["feedback"] = extract_review_feedback(m["context"])
        if cfg.contrastive and cfg.contrastive_mode in (
            "body_derangement",
            "body_recall_guarded",
        ):
            m["entry_point"] = str(rec.get("entry_point", ""))
            m["task_id"] = str(rec.get("task_id", ""))
        records.append(m)
    logger.info("loaded %d mapped training records", len(records))
    # Feedback pool for distribution-matched hard negatives (swap a DIFFERENT row's
    # real feedback into this row's scaffold).
    feedback_pool = (
        [r["feedback"] for r in records if r.get("feedback")]
        if (cfg.contrastive and cfg.contrastive_mode == "feedback_swap_edit_local")
        else []
    )
    val_records: list[dict[str, str]] = []
    if cfg.val_corpus_path:
        val_records = [
            m for rec in _iter_corpus(cfg.val_corpus_path) if (m := _map_record(rec))
        ]
        logger.info("loaded %d held-out val records", len(val_records))

    # body_recall_guarded: freeze the deranged-partner body baseline (lp_n0) under the
    # PRISTINE warm-start hypernet BEFORE any optimizer step, so the guard penalizes the
    # partner RISING above warm-start (anti generic-boost) without rewarding suppress.
    recall_baselines: dict[str, dict[str, Any]] = {}
    if cfg.contrastive and cfg.contrastive_mode == "body_recall_guarded":
        recall_baselines = _precompute_recall_baselines(
            records,
            hypernet,
            base_model,
            tokenizer,
            layer_indices,
            eff_scaling,
            n_chunks,
            cfg.max_seq_length,
        )
        logger.info("precomputed %d recall baselines", len(recall_baselines))
        if cfg.snapshot_steps:  # step-0 warm-start reference for post-hoc deltas
            snap0 = _recall_snapshot(
                records,
                recall_baselines,
                hypernet,
                base_model,
                tokenizer,
                layer_indices,
                eff_scaling,
                n_chunks,
                cfg.snapshot_episodes,
            )
            if mlflow is not None and snap0:
                mlflow.log_metrics(snap0, step=0)
            logger.info("SNAPSHOT step=0 (warm-start) %s", json.dumps(snap0))

    step = 0  # optimizer steps
    micro = 0  # records consumed (micro-steps)
    skipped = 0
    recent_da: list[float] = []
    recent_pres: list[float] = []
    best_val = -1.0  # best held-out val_diff_agreement -> checkpoint_best.pt
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
                    hypernet,
                    context,
                    base_model,
                    tokenizer,
                    layer_indices,
                    cfg.max_seq_length,
                )
                # Assemble a SEPARATE adapter (head bias into ranks r..2r-1 when
                # use_bias); keep raw lora_dict for the L1 term so it never sums the
                # bias ranks. Outside no_grad -> grad flows to weights + head bias.
                assembled = assemble_adapter(hypernet, lora_dict, n_chunks)
                student_logits = _student_logits(
                    base_model,
                    tokenizer,
                    ans_ids,
                    assembled,
                    layer_indices,
                    eff_scaling,
                )

                loss = distill_step_loss(
                    student_logits,
                    teacher_logits,
                    base_top1,
                    teacher_top1,
                    labels,
                    k=cfg.topk,
                )
                student_top1 = student_logits.argmax(dim=-1).detach()
                kl_val = float(loss.detach()) if loss.requires_grad else 0.0
                margin_val = 0.0
                contrastive_metrics = _empty_contrastive_metrics()
                # Contrastive specificity term: matched adapter must beat a hard-neg
                # (feedback-swap edit-local OR body-derangement) on gold tokens.
                # Gradient
                # must flow through BOTH paths — the hinge needs to push the negative
                # adapter's gold logprob DOWN, not only matched UP (else a generic
                # booster lifts both and the gap never opens). Memory-bounded: a
                # detached neg pass fixes the hinge active-set, then the matched piece
                # and neg piece backward sequentially so only one student grad-graph
                # is alive at a time (keeps seq=768, peak ~ single-path).
                neg_ctx: str | None = None
                gold = active = None
                neg_em = None
                em = None
                guard_pending: dict[str, Any] | None = None
                if cfg.contrastive:
                    cmode = cfg.contrastive_mode
                    if cmode == "body_recall_guarded":
                        # PRIMARY (raise matched toward target) + GUARD (hold deranged
                        # at its frozen warm-start baseline). NO suppression reward: the
                        # win can only come from matched up with mismatch held. Guard
                        # piece backprops AFTER the main backward (memory-bounded).
                        loss, contrastive_metrics, guard_pending = _recall_guarded_term(
                            loss=loss,
                            student_logits=student_logits,
                            base_logits=base_logits,
                            ans_ids=ans_ids,
                            answer=answer,
                            mapped=mapped,
                            recall_baselines=recall_baselines,
                            hypernet=hypernet,
                            base_model=base_model,
                            tokenizer=tokenizer,
                            layer_indices=layer_indices,
                            eff_scaling=eff_scaling,
                            n_chunks=n_chunks,
                            cfg=cfg,
                        )
                    elif (
                        cmode == "feedback_swap_edit_local"
                        and feedback_pool
                        and mapped.get("feedback")
                    ):
                        emask = edit_local_mask(
                            tokenizer, mapped.get("pre_code", ""), ans_ids
                        )
                        em = torch.tensor(
                            emask[1:], device=student_logits.device, dtype=torch.bool
                        )
                        neg_ctx = make_hard_negative(
                            context,
                            other_feedback=feedback_pool[micro % len(feedback_pool)],
                        )
                    elif cmode == "body_derangement":
                        # BODY-span contrastive (issue #52): adapter conditioned on this
                        # episode must beat an adapter conditioned on a derangement
                        # partner episode when scoring THIS episode's BODY gold tokens.
                        em = _body_span_mask(
                            tokenizer,
                            answer,
                            mapped.get("entry_point", ""),
                            ans_ids,
                            device=student_logits.device,
                        )
                        neg_ctx = _deranged_partner_context(
                            records, mapped, seed_index=micro
                        )
                    else:
                        em = None
                        neg_ctx = None

                    if neg_ctx is not None and em is not None and int(em.sum()) > 0:
                        gold = torch.tensor(ans_ids[1:], device=student_logits.device)
                        lp_m = (
                            torch.log_softmax(student_logits[:-1].float(), dim=-1)
                            .gather(-1, gold.unsqueeze(-1))
                            .squeeze(-1)[em]
                        )
                        with torch.no_grad():  # detached neg pass: active-set + value
                            neg_ld0 = _generate_lora_dict(
                                hypernet,
                                neg_ctx,
                                base_model,
                                tokenizer,
                                layer_indices,
                                cfg.max_seq_length,
                            )
                            neg_logits0 = _student_logits(
                                base_model,
                                tokenizer,
                                ans_ids,
                                assemble_adapter(hypernet, neg_ld0, n_chunks),
                                layer_indices,
                                eff_scaling,
                            )
                            lp_n_det = (
                                torch.log_softmax(neg_logits0[:-1].float(), dim=-1)
                                .gather(-1, gold.unsqueeze(-1))
                                .squeeze(-1)[em]
                            )
                            contrastive_metrics = _contrastive_logprob_readout(
                                matched_logits=student_logits[:-1],
                                mismatch_logits=neg_logits0[:-1],
                                zero_logits=base_logits[:-1],
                                gold=gold,
                                mask=em,
                                margin=cfg.contrastive_margin,
                            )
                            del neg_ld0, neg_logits0
                        n_tok = int(em.sum())
                        hinge = torch.clamp(
                            cfg.contrastive_margin - (lp_m.detach() - lp_n_det),
                            min=0.0,
                        )
                        margin_val = float(hinge.mean())
                        active = hinge > 0.0
                        if int(active.sum()) > 0:
                            loss = loss + cfg.contrastive_weight * (
                                -(lp_m[active]).sum() / n_tok
                            )
                            neg_em = em  # signal: run neg-grad backward post-matched
                        else:
                            neg_ctx = None
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
                del student_logits  # free matched graph before neg-grad forward
                # Contrastive neg piece (+lp_n on the active set): grad through the
                # negative context so the hinge pushes the wrong-context adapter DOWN.
                if (
                    neg_ctx is not None
                    and neg_em is not None
                    and active is not None
                    and gold is not None
                ):
                    neg_ld = _generate_lora_dict(
                        hypernet,
                        neg_ctx,
                        base_model,
                        tokenizer,
                        layer_indices,
                        cfg.max_seq_length,
                    )
                    neg_logits = _student_logits(
                        base_model,
                        tokenizer,
                        ans_ids,
                        assemble_adapter(hypernet, neg_ld, n_chunks),
                        layer_indices,
                        eff_scaling,
                    )
                    lp_n = (
                        torch.log_softmax(neg_logits[:-1].float(), dim=-1)
                        .gather(-1, gold.unsqueeze(-1))
                        .squeeze(-1)[neg_em]
                    )
                    n_tok = int(neg_em.sum())
                    loss_neg: Any = (
                        cfg.contrastive_weight * (lp_n[active]).sum() / n_tok
                    )
                    (loss_neg / cfg.grad_accum_steps).backward()
                    del neg_ld, neg_logits, lp_n
                # body_recall_guarded GUARD piece: grad through the deranged partner so
                # it held DOWN to its frozen warm-start baseline (relu: no suppression
                # below baseline). Runs only when the partner rose above baseline.
                if guard_pending is not None:
                    g_neg_ld = _generate_lora_dict(
                        hypernet,
                        guard_pending["neg_ctx"],
                        base_model,
                        tokenizer,
                        layer_indices,
                        cfg.max_seq_length,
                    )
                    g_neg_logits = _student_logits(
                        base_model,
                        tokenizer,
                        ans_ids,
                        assemble_adapter(hypernet, g_neg_ld, n_chunks),
                        layer_indices,
                        eff_scaling,
                    )
                    g_lp_n = _gold_logprobs(
                        g_neg_logits[:-1], guard_pending["gold"], guard_pending["em"]
                    )
                    loss_guard: Any = (
                        cfg.guard_weight
                        * torch.clamp(g_lp_n - guard_pending["lp_n0"], min=0.0).mean()
                    )
                    (loss_guard / cfg.grad_accum_steps).backward()
                    del g_neg_ld, g_neg_logits, g_lp_n
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
                        "kl_loss": kl_val,
                        "margin_loss": margin_val,
                        **contrastive_metrics,
                        "ans_len": float(len(ans_ids)),
                        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                        **summarize_named_tensors(watched),
                        **grad_norms,
                    }
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    if mlflow is not None:
                        mlflow.log_metrics(
                            {
                                k: v
                                for k, v in rec.items()
                                if isinstance(v, (int, float))
                            },
                            step=step,
                        )
                    logger.info(
                        "step=%d loss=%.4f diff_agr=%.3f pres=%.3f difftok=%.3f",
                        step,
                        rec["loss"],
                        da,
                        pres,
                        rec["diff_token_frac"],
                    )
                    stop_reason = should_early_stop(
                        step,
                        cfg.early_stop_warmup,
                        recent_da,
                        recent_pres,
                        skipped,
                        skipped + micro,
                        min_diff_agreement=cfg.min_diff_agreement,
                        min_preservation=cfg.min_preservation,
                        max_skip_frac=cfg.max_skip_frac,
                    )
                    if stop_reason is not None:
                        logger.warning("EARLY STOP at step %d: %s", step, stop_reason)
                        if mlflow is not None:
                            mlflow.set_tag("early_stop_reason", stop_reason)
                        break

                # Periodic frozen-probe + generation snapshot (body_recall_guarded): the
                # absent/present BODY lp on the PROBE surface + valid-code gen rate so
                # the objective is tuned against the deciding metric, not answer-only.
                if (
                    did_step
                    and cfg.snapshot_steps
                    and cfg.contrastive_mode == "body_recall_guarded"
                    and step % cfg.snapshot_steps == 0
                ):
                    snap = _recall_snapshot(
                        records,
                        recall_baselines,
                        hypernet,
                        base_model,
                        tokenizer,
                        layer_indices,
                        eff_scaling,
                        n_chunks,
                        cfg.snapshot_episodes,
                    )
                    if mlflow is not None and snap:
                        mlflow.log_metrics(snap, step=step)
                    logger.info("SNAPSHOT step=%d %s", step, json.dumps(snap))

                # Periodic held-out val eval — the ordering/difficulty-independent
                # progress curve (train-loss is confounded by per-row difficulty).
                if did_step and val_records and step % cfg.val_steps == 0:
                    vmet = _eval_on_split(
                        base_model,
                        hypernet,
                        tokenizer,
                        val_records,
                        layer_indices,
                        cfg,
                        hypernet_topk=cfg.topk,
                    )
                    if mlflow is not None:
                        mlflow.log_metrics(vmet, step=step)
                    logger.info("VAL step=%d %s", step, json.dumps(vmet))
                    # Keep the BEST-val checkpoint (anti-overfit: the final-step
                    # checkpoint may be past the val peak).
                    if vmet["val_diff_agreement"] > best_val:
                        best_val = vmet["val_diff_agreement"]
                        _save_checkpoint(
                            hypernet,
                            cfg,
                            step,
                            ckpt_dir,
                            name="checkpoint_best.pt",
                            mlflow_handle=mlflow,
                        )
                        if mlflow is not None:
                            mlflow.set_tag("best_val_diff_agreement", f"{best_val:.4f}")
                            mlflow.set_tag("best_val_step", str(step))

                del lora_dict, teacher_logits, base_logits  # student freed earlier
                gc.collect()
                torch.cuda.empty_cache()

                if did_step and cfg.save_steps and step % cfg.save_steps == 0:
                    # Numbered (kept) checkpoints so the specificity trajectory can
                    # be gated post-hoc — distinguishes "specificity emerges with
                    # training" from "objective is structurally generic".
                    _save_checkpoint(
                        hypernet,
                        cfg,
                        step,
                        ckpt_dir,
                        name=f"checkpoint_step{step}.pt",
                        mlflow_handle=mlflow,
                    )

    _save_checkpoint(hypernet, cfg, step, ckpt_dir, mlflow_handle=mlflow)
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

    from rune.model.adapter_contract import (  # noqa: PLC0415
        assemble_adapter,
        effective_scaling,
    )
    from rune.training.collapse_metrics import (  # noqa: PLC0415
        diff_agreement,
        preservation_agreement,
    )

    eff_scaling = effective_scaling(hypernet)
    n_chunks = torch.ones(1, dtype=torch.int32, device=cfg.device)
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
                base_model,
                tokenizer,
                ans_ids,
                assemble_adapter(hypernet, ld, n_chunks),
                layer_indices,
                eff_scaling,
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
            # bitsandbytes ships no stubs for Adam8bit; route through an Any-typed
            # base so the attribute access + call type-check cleanly whether or not
            # bnb is installed (CI runs CPU-only `uv sync` and treats bnb as Any).
            bnb_any: Any = bnb
            opt8 = bnb_any.optim.Adam8bit
            return opt8(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
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

        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment(cfg.experiment_name)
        mlflow.start_run(run_name=cfg.experiment_name)
        keys = (
            "model_id",
            "corpus_path",
            "learning_rate",
            "num_epochs",
            "max_seq_length",
            "scaler_b_init",
            "train_scaling",
            "topk",
            "l1_reg_coef",
            "max_steps",
            "early_stop_warmup",
            "min_diff_agreement",
            "min_preservation",
            "max_skip_frac",
        )
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
        ans = tt[len(at) :] if tt.startswith(at) else tt
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
                ctx, ans = at, tt[len(at) :]
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
        dist = {
            "min": ans_lens[0],
            "median": ans_lens[n // 2],
            "p90": ans_lens[min(n - 1, int(0.9 * n))],
            "max": ans_lens[-1],
        }
    stats: dict[str, Any] = {
        "raw": raw,
        "mapped": mapped,
        "skipped": raw - mapped,
        "empty_context": empty_ctx,
        "empty_answer": empty_ans,
        "s3_rows": s3_rows,
        "exact_prefix": exact_prefix,
        "fallback": fallback,
        "answer_char_len": dist,
        "fallback_task_ids": fallback_samples,
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
    hypernet: Any,
    cfg: DistillConfig,
    step: int,
    ckpt_dir: Any,
    name: str = "checkpoint.pt",
    mlflow_handle: Any = None,
    keep_local: bool = False,
) -> None:
    """Persist the hypernetwork state dict + config + step to the MLflow (S3)
    artifact store, keeping no local copy.

    The local file is a transient staging path: we torch.save it, upload it via
    MLflow ``log_artifact`` (the server's ``--artifacts-destination`` is S3), verify
    the artifact is present in the store, then delete the local copy. This keeps the
    tiny ~15GB VM disk from filling with multi-hundred-MB checkpoints (the prior
    failure mode). Retrieve later by the ``s3://`` / ``mlflow-artifacts:`` URI —
    ``load_hypernetwork`` downloads + caches s3:// URIs automatically.

    The local file is preserved (and a warning logged) only when upload is impossible
    or unverified — no MLflow handle, ``log_artifact`` raises, or the artifact is not
    found in the store afterward — so a checkpoint is never lost to a failed upload.
    Pass ``keep_local=True`` to force-retain (e.g. a caller that needs the path now).
    """
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
    logger.info("saved checkpoint step=%d → %s (staging)", step, path)

    if mlflow_handle is None:
        logger.warning("no MLflow handle — keeping local checkpoint %s", path)
        return

    artifact_path = "checkpoints"
    try:
        mlflow_handle.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_artifact failed for %s — keeping local: %s", name, exc)
        return

    local_size = path.stat().st_size
    if not keep_local and _artifact_uploaded(
        mlflow_handle, f"{artifact_path}/{name}", local_size
    ):
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(OSError):
            path.unlink()
        logger.info("checkpoint %s on S3 via MLflow; local staging copy removed", name)
    else:
        logger.info("logged checkpoint %s to MLflow (local copy retained)", name)


def _artifact_uploaded(mlflow_handle: Any, rel_path: str, local_size: int) -> bool:
    """Confirm ``rel_path`` is present in the active run's artifact store WITH the
    expected byte size before we delete the local staging file. Path-existence alone is
    insufficient — a reusable name (checkpoint.pt) can list-present while the bytes are
    stale/zero. A False return keeps the local copy."""
    try:
        run = mlflow_handle.active_run()
        if run is None:
            return False
        listed = mlflow_handle.artifacts.list_artifacts(
            run_id=run.info.run_id, artifact_path=rel_path.rsplit("/", 1)[0]
        )
        return any(
            a.path == rel_path and getattr(a, "file_size", None) == local_size
            for a in listed
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not verify artifact %s — keeping local copy: %s", rel_path, exc
        )
        return False


def _deranged_partner_context(
    records: list[dict[str, str]], current: dict[str, str], seed_index: int
) -> str | None:
    """Return a deterministic non-current episode context for contrastive scoring."""
    if len(records) < 2:
        return None

    current_task_id = current.get("task_id")
    current_context = current.get("context")
    start = seed_index % len(records)
    for offset in range(len(records)):
        candidate = records[(start + offset) % len(records)]
        if current_task_id and candidate.get("task_id") == current_task_id:
            continue
        if not current_task_id and candidate.get("context") == current_context:
            continue
        return candidate["context"]
    return None


def _contrastive_logprob_readout(
    *,
    matched_logits: Any,
    mismatch_logits: Any,
    zero_logits: Any,
    gold: Any,
    mask: Any,
    margin: float,
) -> dict[str, float]:
    """Summarize BODY-span contrastive logprobs for trainability diagnostics."""
    import torch  # noqa: PLC0415

    if int(mask.sum()) == 0:
        return _empty_contrastive_metrics()

    lp_matched = _gold_logprobs(matched_logits, gold, mask)
    lp_mismatch = _gold_logprobs(mismatch_logits, gold, mask)
    lp_zero = _gold_logprobs(zero_logits, gold, mask)
    hinge = torch.clamp(margin - (lp_matched - lp_mismatch), min=0.0)
    return {
        "lp_matched": float(lp_matched.mean()),
        "lp_mismatch": float(lp_mismatch.mean()),
        "lp_zero": float(lp_zero.mean()),
        "hinge_active_frac": float((hinge > 0.0).float().mean()),
        "contrastive_tokens": float(int(mask.sum())),
    }


def _empty_contrastive_metrics() -> dict[str, float]:
    """Default contrastive metric values when no contrastive tokens are scored."""
    return {
        "lp_matched": 0.0,
        "lp_mismatch": 0.0,
        "lp_zero": 0.0,
        "hinge_active_frac": 0.0,
        "contrastive_tokens": 0.0,
    }


def _gold_logprobs(logits: Any, gold: Any, mask: Any) -> Any:
    """Gold-token logprobs at masked positions."""
    return (
        logits.float()
        .log_softmax(dim=-1)
        .gather(-1, gold.unsqueeze(-1))
        .squeeze(-1)[mask]
    )


def _body_span_mask(
    tokenizer: Any,
    answer: str,
    entry_point: str,
    ans_ids: list[int],
    *,
    device: Any,
) -> Any:
    """Boolean mask (len == len(ans_ids)-1) selecting BODY gold tokens.

    The scored tokens follow the t-1 convention (student_logits[:-1] predicts
    ans_ids[1:]),
    so the mask indexes the gold-token sequence ans_ids[1:].
    """
    import torch  # noqa: PLC0415

    signature_marker = f"def {entry_point}("
    signature_start = answer.find(signature_marker) if entry_point else -1
    if signature_start < 0:
        raise ValueError(
            "def-<entry_point>( signature marker not found in answer; refusing to "
            "silently score signature as BODY"
        )
    signature_line_end = answer.find("\n", signature_start)
    if signature_line_end < 0:
        signature_line_end = len(answer)
    body_start_token = len(
        tokenizer(answer[: signature_line_end + 1], add_special_tokens=False)[
            "input_ids"
        ]
    )
    # Mask for gold tokens ans_ids[1:], so shift by 1: gold position k corresponds
    # to answer token k+1.
    body_start_gold_index = max(0, body_start_token - 1)
    gold_count = max(0, len(ans_ids) - 1)
    body_start_gold_index = min(body_start_gold_index, gold_count)
    mask = torch.zeros(gold_count, dtype=torch.bool, device=device)
    mask[body_start_gold_index:] = True
    return mask


def _recall_terms(lp_m: Any, lp_n: Any, lp_n0: Any, target: float) -> tuple[Any, Any]:
    """Pure directional terms for body_recall_guarded (testable on CPU).

    primary = relu(target - lp_matched)  -> raises matched ONLY while below target.
    guard   = relu(lp_mismatch - lp_n0)  -> penalizes the deranged partner ONLY when it
              rises ABOVE its frozen warm-start baseline (anti generic-boost); zero when
              held or suppressed below baseline, so there is NO suppression reward.
    """
    import torch  # noqa: PLC0415

    return torch.clamp(target - lp_m, min=0.0), torch.clamp(lp_n - lp_n0, min=0.0)


def _empty_recall_metrics() -> dict[str, float]:
    """Zero-filled body_recall_guarded metrics (inactive/unscored row)."""
    return {
        "lp_matched": 0.0,
        "lp_mismatch": 0.0,
        "lp_zero": 0.0,
        "lp_n0_baseline": 0.0,
        "primary_loss": 0.0,
        "guard_loss": 0.0,
        "primary_active_frac": 0.0,
        "guard_active_frac": 0.0,
        "recall_m_mismatch": 0.0,
        "recall_tokens": 0.0,
    }


def _precompute_recall_baselines(
    records: list[dict[str, str]],
    hypernet: Any,
    base_model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    eff_scaling: float,
    n_chunks: Any,
    max_seq_length: int,
) -> dict[str, dict[str, Any]]:
    """Freeze each episode's deranged-partner BODY logprob baseline under the
    PRISTINE warm-start hypernet (call BEFORE any optimizer step).

    Uses a DETERMINISTIC episode-indexed derangement (``i -> (i+1) % n``, matching the
    frozen E1 probe), so the partner — and thus ``lp_n0`` — is stable across epochs and
    the in-loop guard is consistent with the frozen-probe eval. ``lp_n0[i]`` is the
    per-token gold logprob of episode i's BODY under the partner's adapter.
    """
    import torch  # noqa: PLC0415

    from rune.model.adapter_contract import assemble_adapter  # noqa: PLC0415

    n = len(records)
    baselines: dict[str, dict[str, Any]] = {}
    if n < 2:
        return baselines
    device = next(base_model.parameters()).device
    for i, rec in enumerate(records):
        partner = records[(i + 1) % n]
        tid = rec.get("task_id", "") or f"_idx{i}"
        answer = rec["answer"]
        # ans_ids IDENTICAL to the training loop's (same _teacher_base_logits path).
        _, _, ans_ids = _teacher_base_logits(
            base_model, tokenizer, rec["context"], answer, max_seq_length
        )
        try:
            em = _body_span_mask(
                tokenizer, answer, rec.get("entry_point", ""), ans_ids, device=device
            )
        except ValueError as exc:
            logger.warning("recall baseline skip %s: %s", tid, exc)
            continue
        if int(em.sum()) == 0:
            continue
        gold = torch.tensor(ans_ids[1:], device=device)
        with torch.no_grad():
            neg_ld = _generate_lora_dict(
                hypernet,
                partner["context"],
                base_model,
                tokenizer,
                layer_indices,
                max_seq_length,
            )
            neg_logits = _student_logits(
                base_model,
                tokenizer,
                ans_ids,
                assemble_adapter(hypernet, neg_ld, n_chunks),
                layer_indices,
                eff_scaling,
            )
            lp_n0 = _gold_logprobs(neg_logits[:-1], gold, em).detach()
            del neg_ld, neg_logits
        baselines[tid] = {"lp_n0": lp_n0, "partner_ctx": partner["context"]}
    return baselines


def _recall_guarded_term(
    *,
    loss: Any,
    student_logits: Any,
    base_logits: Any,
    ans_ids: list[int],
    answer: str,
    mapped: dict[str, str],
    recall_baselines: dict[str, dict[str, Any]],
    hypernet: Any,
    base_model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    eff_scaling: float,
    n_chunks: Any,
    cfg: Any,
) -> tuple[Any, dict[str, float], dict[str, Any] | None]:
    """body_recall_guarded contrastive term.

    Returns ``(loss, metrics, guard_pending)``. ``loss`` has the PRIMARY term
    ``lambda_p * relu(target - lp_matched)`` added (grad through matched only).
    ``guard_pending`` (or None) carries the deranged context + frozen ``lp_n0`` for the
    post-backward GUARD piece ``lambda_g * relu(lp_mismatch - lp_n0)`` (grad through the
    partner, held DOWN to its warm-start baseline; relu, so NO suppression below it).
    """
    import torch  # noqa: PLC0415

    from rune.model.adapter_contract import assemble_adapter  # noqa: PLC0415

    tid = mapped.get("task_id", "")
    baseline = recall_baselines.get(tid)
    if baseline is None:
        return loss, _empty_recall_metrics(), None
    try:
        em = _body_span_mask(
            tokenizer,
            answer,
            mapped.get("entry_point", ""),
            ans_ids,
            device=student_logits.device,
        )
    except ValueError:
        return loss, _empty_recall_metrics(), None
    lp_n0 = baseline["lp_n0"]
    if int(em.sum()) == 0 or lp_n0.numel() != int(em.sum()):
        return loss, _empty_recall_metrics(), None

    gold = torch.tensor(ans_ids[1:], device=student_logits.device)
    lp_m = _gold_logprobs(student_logits[:-1], gold, em)  # grad through matched
    target = cfg.matched_target_lp

    neg_ctx = baseline["partner_ctx"]
    with torch.no_grad():  # detached neg pass: readout values + guard active set
        neg_ld0 = _generate_lora_dict(
            hypernet, neg_ctx, base_model, tokenizer, layer_indices, cfg.max_seq_length
        )
        neg_logits0 = _student_logits(
            base_model,
            tokenizer,
            ans_ids,
            assemble_adapter(hypernet, neg_ld0, n_chunks),
            layer_indices,
            eff_scaling,
        )
        lp_n_det = _gold_logprobs(neg_logits0[:-1], gold, em)
        lp_z = _gold_logprobs(base_logits[:-1], gold, em)
        del neg_ld0, neg_logits0
    primary_excess, guard_excess = _recall_terms(lp_m.detach(), lp_n_det, lp_n0, target)

    metrics = {
        "lp_matched": float(lp_m.mean()),
        "lp_mismatch": float(lp_n_det.mean()),
        "lp_zero": float(lp_z.mean()),
        "lp_n0_baseline": float(lp_n0.mean()),
        "primary_loss": float(primary_excess.mean()),
        "guard_loss": float(guard_excess.mean()),
        "primary_active_frac": float((primary_excess > 0.0).float().mean()),
        "guard_active_frac": float((guard_excess > 0.0).float().mean()),
        "recall_m_mismatch": float(lp_m.mean() - lp_n_det.mean()),
        "recall_tokens": float(int(em.sum())),
    }

    if float(primary_excess.sum()) > 0.0:  # raise matched toward target
        loss = loss + cfg.primary_weight * torch.clamp(target - lp_m, min=0.0).mean()
    guard_pending = None
    if float(guard_excess.sum()) > 0.0:  # partner rose above warm-start -> hold it down
        guard_pending = {"neg_ctx": neg_ctx, "em": em, "gold": gold, "lp_n0": lp_n0}
    return loss, metrics, guard_pending


_SNAP_ABSENT = "Write the Python function you have just studied. Return only the code."
_SNAP_PRESENT = "Write the following Python function.\n\n{desc}\n\nReturn only code."
_SNAP_MAX_ANS_TOK = 96


def _snap_full(
    tokenizer: Any, device: Any, prompt: str, answer: str
) -> tuple[Any, int, int]:
    """chat(prompt)+answer ids on the FROZEN-PROBE surface -> (ids, p_len, a_len)."""
    import torch  # noqa: PLC0415

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_special_tokens=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    p = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
    a_ids = tokenizer(answer, add_special_tokens=False).input_ids[:_SNAP_MAX_ANS_TOK]
    # dtype=long: an empty answer ("" for the gen prompt) otherwise yields a float
    # tensor whose cat upcasts the ids -> embedding() rejects float indices.
    a = torch.tensor([a_ids], device=device, dtype=torch.long)
    return torch.cat([p, a], dim=1), p.shape[1], a.shape[1]


def _snap_body_lp(
    logits: Any, ids: Any, start: int, body_start: int, ans_len: int
) -> float:
    """Mean gold logprob over BODY span [start+body_start, start+ans_len) (t-1 conv)."""
    import torch  # noqa: PLC0415

    lp = torch.log_softmax(logits.float(), dim=-1)
    lo, hi = start + body_start, start + ans_len
    if hi <= lo:
        return 0.0
    tot = sum(float(lp[t - 1, ids[t]]) for t in range(lo, hi))
    return tot / (hi - lo)


def _snap_desc(ctx: str) -> str:
    """Recover the task description from a rendered ## Task trajectory context."""
    pre, suf = "## Task\n", "\n\n## Current Code"
    if ctx.startswith(pre) and suf in ctx:
        return ctx[len(pre) : ctx.find(suf)]
    return ctx


def _recall_snapshot(
    records: list[dict[str, str]],
    recall_baselines: dict[str, dict[str, Any]],
    hypernet: Any,
    base_model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    eff_scaling: float,
    n_chunks: Any,
    k: int,
) -> dict[str, float]:
    """Frozen-probe-surface snapshot (the PROBE surface, not the answer-only in-loop
    readout — bridges the engineer-flagged gap). Scores BODY lp matched/mismatch/zero in
    BOTH regimes (absent = operative; present = recitation canary) over k episodes,
    guarded valid-Python generation rate. All under no_grad."""
    import ast  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from rune.engine.graph import render_training_format_trajectory  # noqa: PLC0415
    from rune.model.adapter_contract import assemble_adapter  # noqa: PLC0415

    device = next(base_model.parameters()).device
    n_qs = torch.tensor([1], device=device)
    acc: dict[str, list[float]] = {
        f"{r}_{s}": []
        for r in ("absent", "present")
        for s in ("matched", "mismatch", "zero")
    }
    gen_valid: list[float] = []

    def _lp(adapter: Any, full: Any, ids: Any, start: int, bs: int, al: int) -> float:
        with _functional_lora(base_model, layer_indices, adapter, eff_scaling, n_qs):
            lg = base_model(full, use_cache=False).logits[0]
        return _snap_body_lp(lg, ids, start, bs, al)

    with torch.no_grad():
        for rec in records[:k]:
            tid = rec.get("task_id", "")
            baseline = recall_baselines.get(tid)
            if baseline is None:
                continue
            answer, entry = rec["answer"], rec.get("entry_point", "")
            j = answer.find(f"def {entry}(")
            if j < 0:
                continue
            line_end = answer.find("\n", j)
            line_end = len(answer) if line_end < 0 else line_end
            body_start = len(
                tokenizer(answer[: line_end + 1], add_special_tokens=False).input_ids[
                    :_SNAP_MAX_ANS_TOK
                ]
            )
            desc = _snap_desc(rec["context"])
            traj = render_training_format_trajectory(task=desc)
            m_ld = _generate_lora_dict(
                hypernet, traj, base_model, tokenizer, layer_indices, 2048
            )
            m_ad = assemble_adapter(hypernet, m_ld, n_chunks)
            n_ld = _generate_lora_dict(
                hypernet,
                baseline["partner_ctx"],
                base_model,
                tokenizer,
                layer_indices,
                2048,
            )
            n_ad = assemble_adapter(hypernet, n_ld, n_chunks)
            for regime, tmpl in (("absent", _SNAP_ABSENT), ("present", _SNAP_PRESENT)):
                prompt = tmpl.format(desc=desc) if "{desc}" in tmpl else tmpl
                full, start, al = _snap_full(tokenizer, device, prompt, answer)
                ids = full[0]
                acc[f"{regime}_matched"].append(
                    _lp(m_ad, full, ids, start, body_start, al)
                )
                acc[f"{regime}_mismatch"].append(
                    _lp(n_ad, full, ids, start, body_start, al)
                )
                lz = base_model(full, use_cache=False).logits[0]  # zero = no adapter
                acc[f"{regime}_zero"].append(
                    _snap_body_lp(lz, ids, start, body_start, al)
                )
            # Generation canary (absent, matched): valid code or spec-divergent?
            try:
                full, start, _ = _snap_full(tokenizer, device, _SNAP_ABSENT, "")
                with _functional_lora(
                    base_model, layer_indices, m_ad, eff_scaling, n_qs
                ):
                    gen_out = base_model.generate(
                        full[:, :start],
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(gen_out[0][start:], skip_special_tokens=True)
                try:
                    ast.parse(text)
                    gen_valid.append(1.0)
                except SyntaxError:
                    gen_valid.append(0.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("snapshot gen failed for %s: %s", tid, exc)
            del m_ld, m_ad, n_ld, n_ad

    result: dict[str, float] = {
        f"snap_{key}": sum(vals) / len(vals) for key, vals in acc.items() if vals
    }
    if "snap_absent_matched" in result and "snap_absent_mismatch" in result:
        result["snap_absent_m_mismatch"] = (
            result["snap_absent_matched"] - result["snap_absent_mismatch"]
        )
    if "snap_absent_matched" in result and "snap_absent_zero" in result:
        result["snap_absent_m_zero"] = (
            result["snap_absent_matched"] - result["snap_absent_zero"]
        )
    if gen_valid:
        result["snap_gen_valid_absent"] = sum(gen_valid) / len(gen_valid)
    return result


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
