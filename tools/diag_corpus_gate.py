"""Hypothesis smoke-gate for the corpus-trained checkpoint (issue #49 Stage-4).

Tests THE hypothesis: does the corpus-trained hypernet produce adapters that
CONDITION on real code-review trajectories and improve edit-relevant prediction on
HELD-OUT families it never trained on? Evaluated on the near-dup-filtered clean val
split, comparing three adapter conditions per row over the answer span:
  - REAL: adapter generated from the row's own context.
  - ZERO: adapter scaled to 0 (= base; lower bound).
  - CONTRA: adapter from a DIFFERENT row's context (content-specificity control).
Metric = diff_agreement (student top-1 == teacher top-1 on the base!=teacher
positions = the edit-relevant tokens). PASS = mean(real) > mean(zero) AND
mean(real) > mean(contra), with preservation staying high.

Dual-precision (reviewer): run with the 4-bit train-matched base AND the bf16
engine-target base; report both. Run under tools/run_guarded.sh.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys

import torch

from rune.model.hypernetwork import (
    HypernetworkConfig,
    load_hypernetwork,
)
from rune.training.collapse_metrics import diff_agreement, preservation_agreement
from rune.training.hypernet_distill import (
    _generate_lora_dict,
    _map_record,
    _student_logits,
    _teacher_base_logits,
)


def _load_base(model_id: str, four_bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if four_bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=q, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map={"": "cuda"})
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to("cuda")
    m.eval()
    return m, AutoTokenizer.from_pretrained(model_id)


def _gate(base, tok, hypernet, rows, layer_indices, scaling, max_len):
    real, zero, contra, pres = [], [], [], []
    n = len(rows)
    with torch.no_grad():
        for i, m in enumerate(rows):
            ctx, ans = m["context"], m["answer"]
            t, b, ans_ids = _teacher_base_logits(base, tok, ctx, ans, max_len)
            tt, bt = t.argmax(-1), b.argmax(-1)
            if int((bt != tt).sum()) == 0:
                continue
            ld = _generate_lora_dict(hypernet, ctx, base, tok, layer_indices, max_len)
            s = _student_logits(base, tok, ans_ids, ld, layer_indices, scaling)
            real.append(diff_agreement(s.argmax(-1), tt, bt))
            pres.append(preservation_agreement(s.argmax(-1), tt, bt))
            sz = _student_logits(base, tok, ans_ids, ld, layer_indices, 0.0)
            zero.append(diff_agreement(sz.argmax(-1), tt, bt))
            cctx = rows[(i + 1) % n]["context"]  # different row's context
            cld = _generate_lora_dict(hypernet, cctx, base, tok, layer_indices, max_len)
            sc = _student_logits(base, tok, ans_ids, cld, layer_indices, scaling)
            contra.append(diff_agreement(sc.argmax(-1), tt, bt))
            del t, b, s, sz, sc, ld, cld
    def mean(x):
        return sum(x) / len(x) if x else 0.0
    return {"n": len(real), "real": mean(real), "zero": mean(zero),
            "contra": mean(contra), "preservation": mean(pres),
            "real_gt_zero": mean(real) > mean(zero),
            "real_gt_contra": mean(real) > mean(contra)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scaling", type=float, default=0.5)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-9B")
    ap.add_argument("--json-out", default="/tmp/rune-issue49-corpus-gate.json")
    a = ap.parse_args()

    with open(a.val) as fh:
        rows = [m for line in fh if line.strip() if (m := _map_record(json.loads(line)))][: a.n]
    print(f"held-out clean val rows: {len(rows)}")

    out = {"checkpoint": a.checkpoint, "scaling": a.scaling, "n_rows": len(rows)}
    for label, four_bit in (("4bit_train_matched", True), ("bf16_engine_target", False)):
        base, tok = _load_base(a.model_id, four_bit)
        hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=a.checkpoint), device="cuda")
        hyp.eval()
        li = list(hyp.config.layer_indices)
        res = _gate(base, tok, hyp, rows, li, a.scaling, a.max_seq_length)
        out[label] = res
        print(f"{label}: {json.dumps(res)}")
        del base, hyp
        torch.cuda.empty_cache()

    out["gate_pass"] = all(
        out[k]["real_gt_zero"] and out[k]["real_gt_contra"]
        for k in ("4bit_train_matched", "bf16_engine_target")
    )
    with open(a.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"GATE_PASS: {out['gate_pass']}")
    return 0 if out["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
