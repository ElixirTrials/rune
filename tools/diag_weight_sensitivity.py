"""Does the generated adapter actually depend on the conditioning context? (#49)

The smoke gate showed matched−zero == swapneg−zero == mismatched−zero (+0.86, equal
to 3 decimals). That is consistent with TWO very different stories:
  (a) REAL: the hypernet learned a useful CONSTANT adapter (output-space loss has no
      lever — no contrastive weight can induce specificity), or
  (b) WIRING: extract_activations / generate_weights drops or flattens the context,
      so different inputs map to ~identical weights (a bug, upstream of B1).

This probe (no training, ~5 min) distinguishes them by comparing, for several context
pairs, the EXTRACTED FEATURES and the GENERATED WEIGHTS directly:
  - features differ, weights ~identical  -> generate_weights ignores input (upstream)
  - features ~identical for diff texts    -> extraction is the bottleneck (upstream)
  - weights differ substantially          -> result is REAL, loss has a lever

Pairs: two unrelated rows (diff code+feedback), matched-vs-feedback-swap (same code,
only the review feedback changed = the exact pair B1 must separate), and row-vs-empty.
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


def _flat_weights(lora_dict):
    parts = []
    for w in lora_dict.values():
        parts.append(w["A"].detach().reshape(-1).float())
        parts.append(w["B"].detach().reshape(-1).float())
    return torch.cat(parts)


def _rel(a, b):
    """Relative L2 ||a-b||/||a|| and cosine similarity."""
    num = float(torch.linalg.vector_norm(a - b))
    den = float(torch.linalg.vector_norm(a)) or 1.0
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
    return num / den, cos


def _feat_vec(text, base, tok, li, max_len):
    feats, _ = extract_activations_with_model(
        text=text, model=base, tokenizer=tok, layer_indices=li, max_length=max_len
    )
    return feats.detach().reshape(-1).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/rune-ck-b1smoke/checkpoint_step40.pt")
    ap.add_argument(
        "--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl"
    )
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        a.model_id,
        quantization_config=q,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=a.ckpt), device="cuda")
    hyp.eval()
    li = list(hyp.config.layer_indices)

    rows = []
    with open(a.val) as fh:
        for line in fh:
            if not line.strip():
                continue
            m = _map_record(json.loads(line))
            if m:
                m["feedback"] = extract_review_feedback(m["context"])
                rows.append(m)
            if len(rows) >= 6:
                break

    ctx0 = rows[0]["context"]
    ctx1 = rows[1]["context"]  # unrelated row
    swap = make_hard_negative(
        ctx0, other_feedback=rows[1]["feedback"]
    )  # only feedback changed
    empt = ""

    print(f"ckpt={a.ckpt}")
    print("pair                         feat_rel  feat_cos  |  W_rel    W_cos")
    cases = [
        ("row0 vs row1 (unrelated)", ctx0, ctx1),
        ("row0 vs feedback-swap   ", ctx0, swap),
        ("row0 vs EMPTY            ", ctx0, empt),
    ]
    with torch.no_grad():
        for name, ca, cb in cases:
            fa, fb = (
                _feat_vec(ca, base, tok, li, a.max_seq_length),
                _feat_vec(cb, base, tok, li, a.max_seq_length),
            )
            frel, fcos = _rel(fa, fb)
            wa = _flat_weights(
                _generate_lora_dict(hyp, ca, base, tok, li, a.max_seq_length)
            )
            wb = _flat_weights(
                _generate_lora_dict(hyp, cb, base, tok, li, a.max_seq_length)
            )
            wrel, wcos = _rel(wa, wb)
            print(f"{name}  {frel:7.4f}  {fcos:7.5f}  |  {wrel:7.4f}  {wcos:7.5f}")
            del fa, fb, wa, wb
    print(
        "\nREAD: W_rel ~0 / W_cos ~1 with feat_rel>0 -> generate_weights ignores input "
        "(upstream, no loss fixes it). W_rel substantial -> result REAL, loss has a lever."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
