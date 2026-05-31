"""Decisive H1/H2 experiment for issue #49 (per advisor).

Train the synthetic needle task to SATURATION on two matched records that differ
only in the needle value, then read the forced-choice matched-foil margin:

  under context-A (needle 73921): logprob(73921) - logprob(11111)
  under context-B (needle 11111): logprob(11111) - logprob(73921)

Both margins clearly positive  -> H2: the conditioning pathway works (60 steps was
                                  just too few) -> proceed to real corpus.
Both margins ~ 0 after saturation -> H1: conditioning pathway is broken -> debug it,
                                  don't train more.

This margin is the ONLY metric immune to the generic-perturbation confound: a broad
answer-span perturbation can drop loss / lift diff_agreement but cannot make A prefer
73921 AND B prefer 11111. Cosine / recall-hit / single-needle logprob cannot.

Localizer (matters only if H1): cosine of extracted CONTEXT FEATURES (perceiver
input) A vs C. Features differ but weights stay cosine~1 -> aggregator washes out
context (suspect the eager-attn perceiver patch). Features ~identical -> activation
extraction is the bug.

Run under tools/run_guarded.sh. GPU-only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rune.model.hypernetwork import (
    HypernetworkConfig,
    extract_activations_with_model,
    load_hypernetwork,
    reinit_scaler_b_nonzero,
)
from rune.training.hypernet_distill import (
    _functional_lora,
    _generate_lora_dict,
    _student_logits,
    _teacher_base_logits,
    distill_step_loss,
)

CTX_A = "Internal note: MAGIC_OFFSET = 73921 for the payload."
CTX_B = "Internal note: MAGIC_OFFSET = 11111 for the payload."
CTX_C = "Config: ZORBLAX_LIMIT is set to 48207 in production."
CTX_NEUTRAL = "Internal note: see the configuration file for the payload."
ANS_A = "The MAGIC_OFFSET value is 73921."
ANS_B = "The MAGIC_OFFSET value is 11111."
PREFIX = "The MAGIC_OFFSET value is"
VAL_A = " 73921"
VAL_B = " 11111"
# Held-out generalization pair (NEVER trained): tests a reusable content-binding
# mechanism vs. mere memorization of the A/B prompt-label associations (reviewer).
CTX_D = "Internal note: MAGIC_OFFSET = 55555 for the payload."
VAL_D = " 55555"
VAL_D_FOIL = " 88888"


def _value_logprob(base, tok, hypernet, context, prefix, value, layer_indices, scaling, max_len) -> float:
    """Avg teacher-forced logprob of `value` tokens after `prefix`, adapter from `context`."""
    device = next(base.parameters()).device
    with torch.no_grad():
        ld = _generate_lora_dict(hypernet, context, base, tok, layer_indices, max_len)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        v_ids = tok(value, add_special_tokens=False)["input_ids"]
        full = torch.tensor([p_ids + v_ids], device=device)
        n_qs = torch.tensor([1], device=device)
        ctx = _functional_lora(base, layer_indices, ld, scaling, n_qs) if scaling > 0 else contextlib.nullcontext()
        with ctx:
            logits = base(full, use_cache=False).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        total = sum(float(lp[len(p_ids) + i - 1, tid]) for i, tid in enumerate(v_ids))
        return total / max(len(v_ids), 1)


def _degeneration(base, tok, hypernet, context, prompt, layer_indices, scaling, max_len) -> float:
    device = next(base.parameters()).device
    with torch.no_grad():
        ld = _generate_lora_dict(hypernet, context, base, tok, layer_indices, max_len)
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        n_qs = torch.tensor([1], device=device)
        ctx = _functional_lora(base, layer_indices, ld, scaling, n_qs) if scaling > 0 else contextlib.nullcontext()
        with ctx:
            gen = base.generate(**ids, max_new_tokens=24, do_sample=False)
        new = gen[0, ids["input_ids"].shape[1]:].tolist()
        return (1.0 - len(set(new)) / len(new)) if new else 0.0


def _flat(ld: dict) -> Any:
    return torch.cat([torch.cat([ld[m]["A"].detach().float().reshape(-1),
                                 ld[m]["B"].detach().float().reshape(-1)]) for m in ld])


def _centered_deltas(ld_a: dict, ld_b: dict, ld_neutral: dict) -> dict[str, float]:
    """Centered-delta diagnostics relative to a neutral context (reviewer).

    Raw cosine(W_A, W_B) ~ 1 just means a big shared prior. The behaviorally
    relevant quantity is the CONTEXT delta W(ctx)-W(neutral): its relative norm
    (is context moving the weights at all?) and the cosine between A's and B's
    deltas (do different facts move the weights in DIFFERENT directions?).
    """
    a, b, nz = _flat(ld_a), _flat(ld_b), _flat(ld_neutral)
    nn = nz.norm() + 1e-8
    da, db = a - nz, b - nz
    return {
        "rel_delta_A": float(da.norm() / nn),
        "rel_delta_B": float(db.norm() / nn),
        "centered_delta_cosine_A_vs_B": float(torch.nn.functional.cosine_similarity(da, db, dim=0)),
        "raw_cosine_A_vs_B": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--scaler-b-init", type=float, default=0.1)
    ap.add_argument("--train-scaling", type=float, default=1.0)
    ap.add_argument("--eval-scalings", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--json-out", type=str, default="/tmp/rune-issue49-forced-choice.json")
    ap.add_argument("--checkpoint", type=str,
                    default="s3://elixirtrials-949678234935-eu-west-2-artifacts/checkpoints/hypernet_hpo/checkpoint.pt")
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-9B")
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    hypernet = load_hypernetwork(HypernetworkConfig(checkpoint_path=args.checkpoint), device="cuda")
    reinit_scaler_b_nonzero(hypernet, args.scaler_b_init)
    hypernet.train()
    layer_indices = list(hypernet.config.layer_indices)
    max_len = 256
    out: dict[str, Any] = {"steps": args.steps, "scaler_b_init": args.scaler_b_init,
                           "train_scaling": args.train_scaling}

    # localizer: context-feature cosine A vs C (pre-train, cheap)
    with torch.no_grad():
        fa, _ = extract_activations_with_model(text=CTX_A, model=base, tokenizer=tok,
                                               layer_indices=layer_indices, max_length=max_len)
        fc, _ = extract_activations_with_model(text=CTX_C, model=base, tokenizer=tok,
                                               layer_indices=layer_indices, max_length=max_len)
        n = min(fa.shape[2], fc.shape[2])
        out["feature_cosine_A_vs_C"] = float(
            torch.nn.functional.cosine_similarity(fa[:, :, :n].reshape(-1).float(),
                                                  fc[:, :, :n].reshape(-1).float(), dim=0)
        )

    # train A+B to saturation
    records = [(CTX_A, ANS_A), (CTX_B, ANS_B)]
    opt = torch.optim.AdamW([p for p in hypernet.parameters() if p.requires_grad], lr=args.lr)
    losses = []
    for step in range(args.steps):
        ctx, ans = records[step % 2]
        ld = _generate_lora_dict(hypernet, ctx, base, tok, layer_indices, max_len)
        t, b, ans_ids = _teacher_base_logits(base, tok, ctx, ans, max_len)
        s = _student_logits(base, tok, ans_ids, ld, layer_indices, args.train_scaling)
        lab = torch.ones(t.shape[0], dtype=torch.long, device="cuda")
        loss = distill_step_loss(s, t, b.argmax(-1), t.argmax(-1), lab, k=50)
        if not loss.requires_grad:
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in hypernet.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 50 == 0:
            losses.append({"step": step, "loss": float(loss.detach())})
            print("train", json.dumps(losses[-1]))
        del ld, t, b, s
        torch.cuda.empty_cache()
    out["train_losses"] = losses

    hypernet.eval()
    # centered-delta diagnostics A vs B relative to neutral (post-train)
    with torch.no_grad():
        lda = _generate_lora_dict(hypernet, CTX_A, base, tok, layer_indices, max_len)
        ldb = _generate_lora_dict(hypernet, CTX_B, base, tok, layer_indices, max_len)
        ldn = _generate_lora_dict(hypernet, CTX_NEUTRAL, base, tok, layer_indices, max_len)
        out["centered_deltas"] = _centered_deltas(lda, ldb, ldn)
        out["weight_norm_A"] = float(torch.cat([w["B"].detach().float().reshape(-1) for w in lda.values()]).norm())
        out["weight_norm_B"] = float(torch.cat([w["B"].detach().float().reshape(-1) for w in ldb.values()]).norm())

    # forced-choice margin sweep
    sweep = []
    for s in args.eval_scalings:
        a_correct = _value_logprob(base, tok, hypernet, CTX_A, PREFIX, VAL_A, layer_indices, s, max_len)
        a_foil = _value_logprob(base, tok, hypernet, CTX_A, PREFIX, VAL_B, layer_indices, s, max_len)
        b_correct = _value_logprob(base, tok, hypernet, CTX_B, PREFIX, VAL_B, layer_indices, s, max_len)
        b_foil = _value_logprob(base, tok, hypernet, CTX_B, PREFIX, VAL_A, layer_indices, s, max_len)
        # held-out generalization (unseen value 55555): reusable binding, not memorization
        d_correct = _value_logprob(base, tok, hypernet, CTX_D, PREFIX, VAL_D, layer_indices, s, max_len)
        d_foil = _value_logprob(base, tok, hypernet, CTX_D, PREFIX, VAL_D_FOIL, layer_indices, s, max_len)
        degen = _degeneration(base, tok, hypernet, CTX_A, PREFIX, layer_indices, s, max_len)
        sweep.append({
            "scaling": s,
            "margin_A": a_correct - a_foil,
            "margin_B": b_correct - b_foil,
            "margin_heldout_D": d_correct - d_foil,
            "train_pair_fits": (a_correct - a_foil > 0) and (b_correct - b_foil > 0),
            "heldout_generalizes": (d_correct - d_foil > 0),
            "degeneration": degen,
        })
        print("margin", json.dumps(sweep[-1]))
    out["forced_choice_sweep"] = sweep
    # Downgraded interpretation (reviewer): positive train-pair margins prove the
    # pathway CAN FIT the synthetic pair, not readiness. Held-out positivity is the
    # stronger signal of a reusable content-binding mechanism.
    out["train_pair_fits"] = any(x["train_pair_fits"] and x["degeneration"] < 0.5 for x in sweep)
    out["heldout_generalizes"] = any(x["heldout_generalizes"] and x["degeneration"] < 0.5 for x in sweep)

    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"VERDICT: train_pair_fits={out['train_pair_fits']} "
          f"heldout_generalizes={out['heldout_generalizes']}")
    print("FEATURE_COSINE_A_vs_C:", out["feature_cosine_A_vs_C"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
