"""Recoverability harness (#49): goal / diff / last-N-lines, matched vs zero vs mismatch.

The target an episodic-memory adapter must hit (per the bet): from the adapter ALONE
(episode not in the prompt), the base model should recover —
  - GOAL: the review request driving the episode (## Review Feedback),
  - DIFF: the edit just made (edit-local tokens of the revision),
  - TAIL: the last N lines of the current code = the recent state that DRIVES THE NEXT STEP
    (semi-Markov: given it, the agent picks the next action),
— more than with NO context in the adapter (zero) and more than a MISMATCHED adapter
(a different episode). This is the eval harness for the reformulated data/objective; run
on the current adapter it shows the baseline gap.

Each target: mean gold-token logprob over the target span under
  matched   = adapter from THIS episode's context
  mismatch  = adapter from a DIFFERENT episode's context
  zero      = base, no adapter
Report m-zero (lift over no-adapter) and m-mismatch (episode-specificity). The bet needs
BOTH > 0, especially m-mismatch. No training; base loaded once. Run under run_guarded.sh.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from rune.model.hypernetwork import HypernetworkConfig, load_hypernetwork
from rune.training.contrastive import edit_local_mask, extract_review_feedback
from rune.training.hypernet_distill import (
    _functional_lora,
    _generate_lora_dict,
    _map_record,
)


def _span_logprob(base, tok, hyp, ctx, prompt_ids, target_ids, li, scaling, max_len):
    """Mean gold logprob of target_ids that FOLLOW prompt_ids, under ctx's adapter (or base)."""
    device = next(base.parameters()).device
    if len(target_ids) < 1:
        return None
    ids = torch.tensor([prompt_ids + target_ids], device=device)
    n_qs = torch.tensor([1], device=device)
    with torch.no_grad():
        if scaling > 0:
            ld = _generate_lora_dict(hyp, ctx, base, tok, li, max_len)
            with _functional_lora(base, li, ld, scaling, n_qs):
                logits = base(ids, use_cache=False).logits[0]
            del ld
        else:
            logits = base(ids, use_cache=False).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids)
        tot = sum(float(lp[t - 1, ids[0, t]]) for t in range(start, start + len(target_ids)))
    return tot / len(target_ids)


def _diff_logprob(base, tok, hyp, ctx, answer, pre_code, li, scaling, max_len):
    """Mean gold logprob over EDIT-LOCAL tokens of the (teacher-forced) revision."""
    device = next(base.parameters()).device
    ans_ids = tok(answer, add_special_tokens=False)["input_ids"][:max_len]
    if len(ans_ids) < 2:
        return None
    em = edit_local_mask(tok, pre_code, ans_ids)
    ids = torch.tensor([ans_ids], device=device)
    n_qs = torch.tensor([1], device=device)
    with torch.no_grad():
        if scaling > 0:
            ld = _generate_lora_dict(hyp, ctx, base, tok, li, max_len)
            with _functional_lora(base, li, ld, scaling, n_qs):
                logits = base(ids, use_cache=False).logits[0]
            del ld
        else:
            logits = base(ids, use_cache=False).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        idx = [t for t in range(1, len(ans_ids)) if em[t]]
        if not idx:
            return None
        tot = sum(float(lp[t - 1, ans_ids[t]]) for t in idx)
    return tot / len(idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/rune-ck-final/checkpoint_step600.pt")
    ap.add_argument("--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--scaling", type=float, default=0.5)
    ap.add_argument("--tail-lines", type=int, default=5)
    ap.add_argument("--max-seq-length", type=int, default=768)
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

    rows = []
    with open(a.val) as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line); m = _map_record(raw)
            if not m:
                continue
            rows.append({"ctx": m["context"], "answer": m["answer"],
                         "pre_code": str(raw.get("pre_code", "")),
                         "feedback": extract_review_feedback(m["context"]) or ""})
            if len(rows) >= a.n:
                break

    n = len(rows)
    print(f"ckpt={a.ckpt}  n={n}  scaling={a.scaling}  tail_lines={a.tail_lines}")
    # precompute prompt/target token ids per target
    agg = {t: {"m": [], "x": [], "z": []} for t in ("goal", "diff", "tail")}
    fb_prompt = tok("## Review Feedback\n", add_special_tokens=False)["input_ids"]
    cc_prompt = tok("## Current Code\n", add_special_tokens=False)["input_ids"]
    for i, r in enumerate(rows):
        other = rows[(i + 1) % n]["ctx"]
        # GOAL: recover the review feedback under "## Review Feedback\n"
        if r["feedback"]:
            ft = tok(r["feedback"], add_special_tokens=False)["input_ids"][:128]
            for key, ctx, sc in (("m", r["ctx"], a.scaling), ("x", other, a.scaling), ("z", r["ctx"], 0.0)):
                v = _span_logprob(base, tok, hyp, ctx, fb_prompt, ft, li, sc, a.max_seq_length)
                if v is not None:
                    agg["goal"][key].append(v)
        # DIFF: recover the edit (edit-local tokens of the revision)
        for key, ctx, sc in (("m", r["ctx"], a.scaling), ("x", other, a.scaling), ("z", r["ctx"], 0.0)):
            v = _diff_logprob(base, tok, hyp, ctx, r["answer"], r["pre_code"], li, sc, a.max_seq_length)
            if v is not None:
                agg["diff"][key].append(v)
        # TAIL: last N lines of current code (drives next step) given the earlier code
        lines = r["pre_code"].splitlines(keepends=True)
        if len(lines) > a.tail_lines + 2:
            prefix = "".join(lines[:-a.tail_lines])
            tail = "".join(lines[-a.tail_lines:])
            prompt_ids = cc_prompt + tok(prefix, add_special_tokens=False)["input_ids"][-(a.max_seq_length - 160):]
            tt = tok(tail, add_special_tokens=False)["input_ids"][:128]
            for key, ctx, sc in (("m", r["ctx"], a.scaling), ("x", other, a.scaling), ("z", r["ctx"], 0.0)):
                v = _span_logprob(base, tok, hyp, ctx, prompt_ids, tt, li, sc, a.max_seq_length)
                if v is not None:
                    agg["tail"][key].append(v)

    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    print(f"\n{'target':6} (n)   matched   mismatch  zero    | m-mismatch  m-zero")
    for t in ("goal", "diff", "tail"):
        mm, xx, zz = mean(agg[t]["m"]), mean(agg[t]["x"]), mean(agg[t]["z"])
        print(f"{t:6} ({len(agg[t]['m']):2})  {mm:8.4f}  {xx:8.4f}  {zz:8.4f} | {mm-xx:+.5f}   {mm-zz:+.4f}")
    print("\nREAD: the bet needs m-mismatch>0 (episode-specific) AND m-zero>0 (beats no-context). "
          "tail = the semi-Markov 'drives next step' signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
