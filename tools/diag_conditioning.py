"""Centered/layerwise conditioning probe (#49) — does the context RESIDUAL exist?

Refines diag_weight_sensitivity per reflections: a single global cosine is dominated
by the shared generic adapter and hides a small high-leverage residual. Here we
decompose each generated adapter as W(ctx) = W_mean + residual(ctx) over a set of
contexts, and report:
  - per-context generated-weight norm ||W(ctx)||
  - centered residual ||W(ctx) - W_mean|| and its ratio to ||W_mean|| (the fraction
    of the adapter that is context-DEPENDENT vs shared-generic)
  - feature centered residual (same decomposition on extract_activations output)
  - layerwise relative L2 for matched-vs-swap and matched-vs-mismatch (which layer,
    if any, carries a context delta)

Decision:
  - residual ratio ~0 (weights ~constant)        -> conditioning/representation failure
  - residual real but edit-logprob flat @0.5      -> scale / rank / generic swallows it
    (-> Sakana up-scaling: re-gate across 0.25..2.0)
No training; loads base once. Run under tools/run_guarded.sh.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from rune.model.hypernetwork import (
    HypernetworkConfig,
    extract_activations_with_model,
    load_hypernetwork,
)
from rune.training.contrastive import extract_review_feedback, make_hard_negative
from rune.training.hypernet_distill import _generate_lora_dict, _map_record


def _flat(lora_dict):
    parts = []
    for w in lora_dict.values():
        parts.append(w["A"].detach().reshape(-1).float())
        parts.append(w["B"].detach().reshape(-1).float())
    return torch.cat(parts)


def _per_layer(lora_dict, n_layers):
    """Return [n_layers] list of flat per-layer weight vectors (all modules concat)."""
    out = []
    for li in range(n_layers):
        parts = []
        for w in lora_dict.values():
            parts.append(w["A"][0, li].detach().reshape(-1).float())
            parts.append(w["B"][0, li].detach().reshape(-1).float())
        out.append(torch.cat(parts))
    return out


def _relL2(a, b):
    return float(torch.linalg.vector_norm(a - b)) / (float(torch.linalg.vector_norm(a)) or 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/rune-ck-b1smoke/checkpoint_step40.pt")
    ap.add_argument("--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl")
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--n-ctx", type=int, default=5, help="distinct rows for the residual set")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        a.model_id, quantization_config=q, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": "cuda"}).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=a.ckpt), device="cuda")
    hyp.eval(); li = list(hyp.config.layer_indices)
    n_layers = len(li)

    rows = []
    with open(a.val) as fh:
        for line in fh:
            if not line.strip():
                continue
            m = _map_record(json.loads(line))
            if m:
                m["feedback"] = extract_review_feedback(m["context"])
                rows.append(m)
            if len(rows) >= a.n_ctx:
                break

    # context set: the n distinct rows + empty; plus swap(row0) for the matched/swap pair
    ctxs = [r["context"] for r in rows] + [""]
    labels = [f"row{i}" for i in range(len(rows))] + ["EMPTY"]
    swap0 = make_hard_negative(rows[0]["context"], other_feedback=rows[1]["feedback"])

    with torch.no_grad():
        feats = [extract_activations_with_model(text=c, model=base, tokenizer=tok,
                 layer_indices=li, max_length=a.max_seq_length)[0].detach().reshape(-1).float()
                 for c in ctxs]
        weights = [_flat(_generate_lora_dict(hyp, c, base, tok, li, a.max_seq_length)) for c in ctxs]
        w_swap = _flat(_generate_lora_dict(hyp, swap0, base, tok, li, a.max_seq_length))

        f_mean = torch.stack(feats).mean(0)
        w_mean = torch.stack(weights).mean(0)
        print(f"ckpt={a.ckpt}  n_layers={n_layers}  n_ctx={len(ctxs)}")
        print(f"||W_mean||={float(torch.linalg.vector_norm(w_mean)):.4f}  "
              f"||feat_mean||={float(torch.linalg.vector_norm(f_mean)):.4f}")
        print("\nctx       ||W||     ||W-Wmean||  resid/Wmean | ||feat||  feat_resid/featmean")
        for lab, w, f in zip(labels, weights, feats):
            wr = float(torch.linalg.vector_norm(w - w_mean))
            wn = float(torch.linalg.vector_norm(w))
            fr = float(torch.linalg.vector_norm(f - f_mean))
            fn = float(torch.linalg.vector_norm(f))
            print(f"{lab:8} {wn:9.4f} {wr:10.5f}  {wr/(float(torch.linalg.vector_norm(w_mean)) or 1):9.5f}  | "
                  f"{fn:8.3f}  {fr/(float(torch.linalg.vector_norm(f_mean)) or 1):.5f}")

        # the B1-critical pair: matched(row0) vs feedback-swap(row0) — ONLY feedback differs
        print(f"\nmatched(row0) vs feedback-SWAP: global W relL2 = {_relL2(weights[0], w_swap):.6f}  "
              f"feat relL2 = {_relL2(feats[0], extract_activations_with_model(text=swap0, model=base, tokenizer=tok, layer_indices=li, max_length=a.max_seq_length)[0].detach().reshape(-1).float()):.6f}")
        # layerwise for matched vs swap and matched vs row1(mismatch)
        ld0 = _generate_lora_dict(hyp, rows[0]["context"], base, tok, li, a.max_seq_length)
        lds = _generate_lora_dict(hyp, swap0, base, tok, li, a.max_seq_length)
        ld1 = _generate_lora_dict(hyp, rows[1]["context"], base, tok, li, a.max_seq_length)
        pl0, pls, pl1 = _per_layer(ld0, n_layers), _per_layer(lds, n_layers), _per_layer(ld1, n_layers)
        sw = [_relL2(pl0[L], pls[L]) for L in range(n_layers)]
        mm = [_relL2(pl0[L], pl1[L]) for L in range(n_layers)]
        top_sw = sorted(range(n_layers), key=lambda L: sw[L], reverse=True)[:5]
        print("layerwise relL2 (matched vs swap)   max@layers:",
              [(li[L], round(sw[L], 5)) for L in top_sw])
        top_mm = sorted(range(n_layers), key=lambda L: mm[L], reverse=True)[:5]
        print("layerwise relL2 (matched vs row1)   max@layers:",
              [(li[L], round(mm[L], 5)) for L in top_mm])
    print("\nREAD: resid/Wmean ~0 across rows -> ~constant adapter (conditioning failure). "
          "Substantial resid but flat edit-logprob @0.5 -> scale/rank/generic-swallow (Sakana up-scale).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
