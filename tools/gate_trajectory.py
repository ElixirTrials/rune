"""Specificity-trajectory gate across checkpoints × scalings (issue #49 campaign).

THE analysis run when a training experiment completes. For each checkpoint and each
adapter scaling, evaluates the adapter-as-memory signal on the clean held-out val
split: edit-local logprob of the gold revision under MATCHED vs MISMATCHED vs ZERO
adapters (canonical edit_local_mask, shared with contrastive training).

Reports, per (checkpoint, scaling): matched, mismatched, zero, matched−mismatched
(the memory signal), matched−zero (generic lift), preservation. The TRAJECTORY of
matched−mismatched across checkpoints distinguishes "specificity emerges with
training" (rises) from "objective is generic" (flat while matched−zero rises).

Loads the base ONCE (4-bit, train-matched). Run under tools/run_guarded.sh.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import torch

from rune.model.hypernetwork import HypernetworkConfig, load_hypernetwork
from rune.training.collapse_metrics import diff_agreement, preservation_agreement
from rune.training.contrastive import (
    edit_local_mask,
    extract_review_feedback,
    make_hard_negative,
)
from rune.training.hypernet_distill import (
    _generate_lora_dict,
    _student_logits,
    _teacher_base_logits,
)


def _editlocal_logprob(base, tok, hyp, ctx, ans, pre_code, li, scaling, max_len):
    """Mean next-token logprob of the gold edit on EDIT-LOCAL tokens.

    Uses the same answer-only student forward as training (_student_logits): the
    adapter carries the context, the prompt is the answer span only. scaling==0 is
    the zero-adapter (base) baseline.
    """
    device = next(base.parameters()).device
    ans_ids = tok(ans, add_special_tokens=False)["input_ids"][:max_len]
    if len(ans_ids) < 2:
        return None
    emask = edit_local_mask(tok, pre_code, ans_ids)
    with torch.no_grad():
        if scaling > 0:
            ld = _generate_lora_dict(hyp, ctx, base, tok, li, max_len)
            logits = _student_logits(base, tok, ans_ids, ld, li, scaling)
            del ld
        else:
            ids = torch.tensor([ans_ids], device=device)
            logits = base(ids, use_cache=False).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tot = cnt = 0
        for t in range(1, len(ans_ids)):
            if emask[t]:
                tot += float(lp[t - 1, ans_ids[t]])
                cnt += 1
    return tot / cnt if cnt else None


def _row_specificity(base, tok, hyp, rows, li, scaling, max_len):
    """Edit-local logprob under MATCHED / MISMATCHED / SWAP-NEG / ZERO adapters.

    Three negatives are distinguished (advisor + reflections): MISMATCHED = a
    different row's context (transfer test); SWAP-NEG = the same row with feedback
    swapped from another row (the EXACT negative B1 trains on); ZERO = no adapter.
    Decomposing matched−zero (generic lift), swapneg−zero (inappropriate neg lift,
    should DROP after contrastive), matched−swapneg (the trained objective) and
    matched−mismatched (transfer memory) tells WHY a margin opened.
    """
    m, x, sn, z, da, pres = [], [], [], [], [], []
    n = len(rows)
    for i, r in enumerate(rows):
        other_fb = rows[(i + 1) % n].get("feedback", "")
        swap_ctx = make_hard_negative(r["context"], other_feedback=other_fb)
        lm = _editlocal_logprob(
            base,
            tok,
            hyp,
            r["context"],
            r["answer"],
            r["pre_code"],
            li,
            scaling,
            max_len,
        )
        lx = _editlocal_logprob(
            base,
            tok,
            hyp,
            rows[(i + 1) % n]["context"],
            r["answer"],
            r["pre_code"],
            li,
            scaling,
            max_len,
        )
        ls = _editlocal_logprob(
            base, tok, hyp, swap_ctx, r["answer"], r["pre_code"], li, scaling, max_len
        )
        lz = _editlocal_logprob(
            base, tok, hyp, r["context"], r["answer"], r["pre_code"], li, 0.0, max_len
        )
        if None in (lm, lx, ls, lz):
            continue
        m.append(lm)
        x.append(lx)
        sn.append(ls)
        z.append(lz)
        # diff_agreement/preservation at this scaling (top-1 based)
        t, b, ans_ids = _teacher_base_logits(
            base, tok, r["context"], r["answer"], max_len
        )
        tt, bt = t.argmax(-1), b.argmax(-1)
        if int((bt != tt).sum()) > 0:
            ld = _generate_lora_dict(hyp, r["context"], base, tok, li, max_len)
            s = _student_logits(base, tok, ans_ids, ld, li, scaling).argmax(-1)
            da.append(diff_agreement(s, tt, bt))
            pres.append(preservation_agreement(s, tt, bt))
            del ld
        del t, b
    mean = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731
    return {
        "n": len(m),
        "matched": mean(m),
        "mismatched": mean(x),
        "swapneg": mean(sn),
        "zero": mean(z),
        "matched_minus_mismatched": mean(m) - mean(x),
        "matched_minus_swapneg": mean(m) - mean(sn),
        "matched_minus_zero": mean(m) - mean(z),
        "swapneg_minus_zero": mean(sn) - mean(z),
        "diff_agreement": mean(da),
        "preservation": mean(pres),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/tmp/rune-ck-final")
    ap.add_argument(
        "--ckpts", nargs="*", default=None, help="explicit checkpoint paths"
    )
    ap.add_argument(
        "--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl"
    )
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--scalings", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--json-out", default="/tmp/rune-issue49-trajectory-gate.json")
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
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)

    rows = []
    with open(a.val) as fh:
        from rune.training.hypernet_distill import _map_record

        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            mm = _map_record(raw)
            if mm:
                rows.append(
                    {
                        **mm,
                        "pre_code": str(raw.get("pre_code", "")),
                        "feedback": extract_review_feedback(mm["context"]),
                    }
                )
            if len(rows) >= a.n:
                break

    ckpts = a.ckpts or sorted(
        glob.glob(f"{a.ckpt_dir}/checkpoint_step*.pt"),
        key=lambda p: int(p.split("step")[-1].split(".")[0]),
    ) + glob.glob(f"{a.ckpt_dir}/checkpoint_best.pt")
    out = {"ckpt_dir": a.ckpt_dir, "results": []}
    for ck in ckpts:
        hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=ck), device="cuda")
        hyp.eval()
        li = list(hyp.config.layer_indices)
        for s in a.scalings:
            r = _row_specificity(base, tok, hyp, rows, li, s, a.max_seq_length)
            r["ckpt"] = ck.split("/")[-1]
            r["scaling"] = s
            out["results"].append(r)
            print(
                f"{r['ckpt']:22} sc={s}: m-mm={r['matched_minus_mismatched']:+.4f} "
                f"m-swap={r['matched_minus_swapneg']:+.4f} m-zero={r['matched_minus_zero']:+.3f} "
                f"swap-zero={r['swapneg_minus_zero']:+.3f} pres={r['preservation']:.3f} n={r['n']}"
            )
        del hyp
        torch.cuda.empty_cache()
    with open(a.json_out, "w") as f:
        json.dump(out, f, indent=2)
    # trajectory verdict: does matched−mismatched rise across checkpoints (any scaling)?
    print("TRAJECTORY-GATE done →", a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
