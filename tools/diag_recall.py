"""Do the adapters present RECOVERABLE FACTS? First-principles recall test (#49).

Sakana/doc2lora definition of adapter-as-memory: base+W(ctx) recovers information
about ctx that base alone cannot, and that base+W(other_ctx) cannot. Our supervised
target (the revision) is ~89% a verbatim COPY of the context's Current Code, with a
~10% edit. So we split the answer tokens:
  - COPY tokens  (answer ⊂ pre_code): the row's SPECIFIC code body. Pure recall: does
    the adapter store THIS row's code? (uncontaminated by generic edit-boosting)
  - EDIT tokens  (the diff): the requested change (confounded with generic editing)
  - FULL: all answer tokens.

For each slice we report mean gold-token logprob under MATCHED vs MISMATCHED (a
different row's adapter) vs ZERO (base, no adapter). The decisive number is
COPY matched-mismatched: >0 means the adapter recalls the row's specific code
(recoverable facts EXIST); ~0 means the adapter encodes no recoverable
context-specific facts at all -- the most basic failure, upstream of any contrast.
No training; base loaded once. Run under tools/run_guarded.sh.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from rune.model.hypernetwork import HypernetworkConfig, load_hypernetwork
from rune.training.contrastive import edit_local_mask
from rune.training.hypernet_distill import (
    _generate_lora_dict,
    _map_record,
    _student_logits,
)


def _slice_logprobs(base, tok, hyp, ctx, ans, pre_code, li, scaling, max_len):
    """Return (full, copy, edit) mean gold logprob for one (ctx-adapter, answer)."""
    device = next(base.parameters()).device
    ans_ids = tok(ans, add_special_tokens=False)["input_ids"][:max_len]
    if len(ans_ids) < 2:
        return None
    emask = edit_local_mask(tok, pre_code, ans_ids)  # True = edit token
    with torch.no_grad():
        if scaling > 0:
            ld = _generate_lora_dict(hyp, ctx, base, tok, li, max_len)
            logits = _student_logits(base, tok, ans_ids, ld, li, scaling)
            del ld
        else:
            logits = base(
                torch.tensor([ans_ids], device=device), use_cache=False
            ).logits[0]
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        gold = torch.tensor(ans_ids[1:], device=device)
        tok_lp = lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)  # [T-1]
        em = torch.tensor(emask[1:], device=device, dtype=torch.bool)
        full = float(tok_lp.mean())
        edit = float(tok_lp[em].mean()) if int(em.sum()) else float("nan")
        copy = float(tok_lp[~em].mean()) if int((~em).sum()) else float("nan")
    return full, copy, edit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/rune-ck-b1smoke/checkpoint_step40.pt")
    ap.add_argument(
        "--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl"
    )
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--scalings", type=float, nargs="+", default=[0.5, 1.0])
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
                m["pre_code"] = ""  # filled below from raw
                rows.append((m, json.loads(line)))
            if len(rows) >= a.n:
                break
    for m, raw in rows:
        m["pre_code"] = str(raw.get("pre_code", ""))

    print(f"ckpt={a.ckpt}  n={len(rows)}")
    n = len(rows)
    for s in a.scalings:
        agg = {k: {"m": [], "x": [], "z": []} for k in ("full", "copy", "edit")}
        for i, (m, _) in enumerate(rows):
            xm = rows[(i + 1) % n][0]["context"]  # mismatched: different row's context
            rm = _slice_logprobs(
                base,
                tok,
                hyp,
                m["context"],
                m["answer"],
                m["pre_code"],
                li,
                s,
                a.max_seq_length,
            )
            rx = _slice_logprobs(
                base, tok, hyp, xm, m["answer"], m["pre_code"], li, s, a.max_seq_length
            )
            rz = _slice_logprobs(
                base,
                tok,
                hyp,
                m["context"],
                m["answer"],
                m["pre_code"],
                li,
                0.0,
                a.max_seq_length,
            )
            if None in (rm, rx, rz):
                continue
            for j, k in enumerate(("full", "copy", "edit")):
                if rm[j] == rm[j]:  # not nan
                    agg[k]["m"].append(rm[j])
                    agg[k]["x"].append(rx[j])
                    agg[k]["z"].append(rz[j])
        mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
        print(f"\n-- scaling={s} --")
        print(f"{'slice':5}  matched   mismatch  zero    | m-mismatch  m-zero")
        for k in ("full", "copy", "edit"):
            mm, xx, zz = mean(agg[k]["m"]), mean(agg[k]["x"]), mean(agg[k]["z"])
            print(
                f"{k:5}  {mm:8.4f}  {xx:8.4f}  {zz:8.4f} | {mm - xx:+.5f}   {mm - zz:+.4f}"
            )
    print(
        "\nREAD: COPY m-mismatch >0 -> adapter recalls THIS row's code (recoverable facts). "
        "~0 -> no recoverable context-specific facts (adapter is context-invariant memory)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
