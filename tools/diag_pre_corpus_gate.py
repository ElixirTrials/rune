"""Pre-corpus robustness gate for issue #49 (per reviewer + user option 1).

Hardens the H2 go-signal before committing to real-corpus training:
  1. MULTI-VALUE held-out generalization: train a 2-fact synthetic pair, then test
     forced-choice margins on SEVERAL unseen values (matched 5-digit tokenization),
     each against multiple neutral distractor foils (worst-case margin).
  2. VARIANCE across configs: repeat with a different training value pair + seed, so
     the signal is not specific to one pair/RNG.
  3. PEFT train/inference PARITY smoke: same generated lora_dict applied via the
     training functional path vs the exported PEFT hot-swap path at matched scaling;
     assert logits agree. Guards against optimizing a path the engine cannot deploy.

Pre-registered diagnostic = centered-delta vs neutral (raw cosine is non-decisive).

Run under tools/run_guarded.sh. GPU-only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from rune.model.adapter import scale_lora_b
from rune.model.hypernetwork import (
    HypernetworkConfig,
    _to_peft_state_dict,
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

PREFIX = "The MAGIC_OFFSET value is"
CONFIGS = [
    {"seed": 0, "train": [73921, 11111], "holdouts": [55555, 31415, 27182, 90210]},
    {"seed": 1, "train": [42042, 86753], "holdouts": [12345, 98765, 24680, 13579]},
]
DISTRACTORS = [88888, 33333, 64646]  # neutral foils never used as a context value


def _ctx(v: int) -> str:
    return f"Internal note: MAGIC_OFFSET = {v} for the payload."


def _ans(v: int) -> str:
    return f"The MAGIC_OFFSET value is {v}."


def _val_logprob(base, tok, hypernet, context, value, layer_indices, scaling, max_len) -> float:
    device = next(base.parameters()).device
    with torch.no_grad():
        ld = _generate_lora_dict(hypernet, context, base, tok, layer_indices, max_len)
        p_ids = tok(PREFIX, add_special_tokens=False)["input_ids"]
        v_ids = tok(f" {value}", add_special_tokens=False)["input_ids"]
        full = torch.tensor([p_ids + v_ids], device=device)
        n_qs = torch.tensor([1], device=device)
        ctx = _functional_lora(base, layer_indices, ld, scaling, n_qs) if scaling > 0 else contextlib.nullcontext()
        with ctx:
            logits = base(full, use_cache=False).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        return sum(float(lp[len(p_ids) + i - 1, t]) for i, t in enumerate(v_ids)) / max(len(v_ids), 1)


def _margin(base, tok, hypernet, value, foils, layer_indices, scaling, max_len) -> float:
    """Forced-choice worst-case margin: lp(correct) - max_foil lp(foil), under value's context."""
    ctx = _ctx(value)
    correct = _val_logprob(base, tok, hypernet, ctx, value, layer_indices, scaling, max_len)
    worst_foil = max(_val_logprob(base, tok, hypernet, ctx, f, layer_indices, scaling, max_len) for f in foils)
    return correct - worst_foil


def _train(base, tok, hypernet, pairs, layer_indices, train_scaling, steps, lr, max_len) -> list:
    opt = torch.optim.AdamW([p for p in hypernet.parameters() if p.requires_grad], lr=lr)
    losses = []
    recs = [(_ctx(v), _ans(v)) for v in pairs]
    for step in range(steps):
        ctx, ans = recs[step % len(recs)]
        ld = _generate_lora_dict(hypernet, ctx, base, tok, layer_indices, max_len)
        t, b, sl = _teacher_base_logits(base, tok, ctx, ans, max_len)
        s = _student_logits(base, tok, ans, ld, sl, layer_indices, train_scaling)
        lab = torch.ones(t.shape[0], dtype=torch.long, device=next(base.parameters()).device)
        loss = distill_step_loss(s, t, b.argmax(-1), t.argmax(-1), lab, k=50)
        if not loss.requires_grad:
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in hypernet.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 100 == 0:
            losses.append({"step": step, "loss": float(loss.detach())})
        del ld, t, b, s
        torch.cuda.empty_cache()
    return losses


def _peft_parity(base, tok, hypernet, layer_indices, scaling, max_len) -> dict:
    """Compare functional-path logits vs PEFT export+hotswap logits on the same lora_dict."""
    device = next(base.parameters()).device
    hc = hypernet.config
    rank = hc.lora_config.r
    target_modules = list(hc.lora_config.target_modules)
    probe = "def add(a, b):\n    return"
    ids = tok(probe, add_special_tokens=False, return_tensors="pt").to(device)

    with torch.no_grad():
        ld = _generate_lora_dict(hypernet, _ctx(73921), base, tok, layer_indices, max_len)
        n_qs = torch.tensor([1], device=device)
        with _functional_lora(base, layer_indices, ld, scaling, n_qs):
            func_logits = base(**ids, use_cache=False).logits[0, -1].float()

        # PEFT path: alpha=r -> scaling alpha/r=1; scale_lora_b by `scaling` to match.
        peft = get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules,
                                               lora_dropout=0.0, task_type="CAUSAL_LM"))
        sd = scale_lora_b(_to_peft_state_dict(ld, layer_indices, target_modules), scaling)
        set_peft_model_state_dict(peft, sd)
        peft_logits = peft(**ids, use_cache=False).logits[0, -1].float()

    diff = (func_logits - peft_logits).abs().max().item()
    cos = float(torch.nn.functional.cosine_similarity(func_logits, peft_logits, dim=0))
    return {"max_abs_logit_diff": diff, "cosine": cos, "pass": diff < 0.5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--scaler-b-init", type=float, default=0.1)
    ap.add_argument("--train-scaling", type=float, default=1.0)
    ap.add_argument("--eval-scalings", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--json-out", type=str, default="/tmp/rune-issue49-pre-corpus.json")
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
    max_len = 256
    out: dict[str, Any] = {"configs": []}

    for cfg in CONFIGS:
        torch.manual_seed(cfg["seed"])
        hypernet = load_hypernetwork(HypernetworkConfig(checkpoint_path=args.checkpoint), device="cuda")
        reinit_scaler_b_nonzero(hypernet, args.scaler_b_init)
        hypernet.train()
        layer_indices = list(hypernet.config.layer_indices)
        losses = _train(base, tok, hypernet, cfg["train"], layer_indices,
                        args.train_scaling, args.steps, args.lr, max_len)
        hypernet.eval()

        per_scaling = []
        for s in args.eval_scalings:
            # train-pair margins (each trained value vs the other trained value as foil)
            tp = [_margin(base, tok, hypernet, v, [o for o in cfg["train"] if o != v] + DISTRACTORS,
                          layer_indices, s, max_len) for v in cfg["train"]]
            # held-out margins (each unseen value vs distractors + other held-outs)
            ho = []
            for v in cfg["holdouts"]:
                foils = DISTRACTORS + [h for h in cfg["holdouts"] if h != v]
                ho.append(_margin(base, tok, hypernet, v, foils, layer_indices, s, max_len))
            per_scaling.append({
                "scaling": s,
                "train_pair_margins": tp,
                "train_pair_min": min(tp),
                "holdout_margins": ho,
                "holdout_min": min(ho),
                "holdout_median": sorted(ho)[len(ho) // 2],
                "holdout_frac_positive": sum(m > 0 for m in ho) / len(ho),
            })
        out["configs"].append({"seed": cfg["seed"], "train": cfg["train"],
                               "final_loss": losses[-1]["loss"] if losses else None,
                               "per_scaling": per_scaling})
        print(f"config seed={cfg['seed']}:", json.dumps(per_scaling))

    # PEFT parity smoke (uses last config's trained hypernet)
    out["peft_parity"] = _peft_parity(base, tok, hypernet, layer_indices, 1.0, max_len)
    print("PEFT_PARITY:", json.dumps(out["peft_parity"]))

    # Aggregate verdict: held-out generalizes robustly if, at some non-degenerate
    # scaling, every config has >=75% held-out values with positive margin; parity must pass.
    robust = all(
        any(ps["holdout_frac_positive"] >= 0.75 for ps in c["per_scaling"])
        for c in out["configs"]
    )
    out["heldout_robust"] = robust
    out["parity_pass"] = out["peft_parity"]["pass"]
    out["gate_pass"] = robust and out["parity_pass"]
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"GATE: heldout_robust={robust} parity_pass={out['parity_pass']} gate_pass={out['gate_pass']}")
    return 0 if out["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
