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
import difflib
import json
import sys

import torch

from rune.config import load_rune_config
from rune.model.adapter_contract import assemble_adapter, effective_scaling
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
            ld = assemble_adapter(hyp, ld, n_qs)  # head bias -> ranks r..2r-1
            with _functional_lora(base, li, ld, scaling, n_qs):
                logits = base(ids, use_cache=False).logits[0]
            del ld
        else:
            logits = base(ids, use_cache=False).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids)
        tot = sum(
            float(lp[t - 1, ids[0, t]]) for t in range(start, start + len(target_ids))
        )
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
            ld = assemble_adapter(hyp, ld, n_qs)  # head bias -> ranks r..2r-1
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


def _avoid_failed_margin(
    base, tok, hyp, ctx, pre_code, post_code, li, scaling, max_len
):
    """logp(accepted post-form) - logp(rejected pre-form) at the first changed hunk.

    The pre-edit form is what the reviewer REJECTED (the 'tried and failed' approach); the
    post-edit form is the accepted fix. A 'don't repeat the mistake' memory should prefer
    the accepted over the rejected continuation after the shared prefix. Returns the margin
    (higher = avoids the rejected form). None if there is no replaced hunk.
    """
    sm = difflib.SequenceMatcher(None, pre_code, post_code)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i2 > i1 and j2 > j1:
            prefix = post_code[
                max(0, j1 - 400) : j1
            ]  # shared accepted code just before the edit
            pre_form = pre_code[i1:i2]  # rejected
            post_form = post_code[j1:j2]  # accepted
            break
    else:
        return None
    p_ids = tok("## Current Code\n" + prefix, add_special_tokens=False)["input_ids"][
        -(max_len - 96) :
    ]
    pre_ids = tok(pre_form, add_special_tokens=False)["input_ids"][:64]
    post_ids = tok(post_form, add_special_tokens=False)["input_ids"][:64]
    lp_pre = _span_logprob(base, tok, hyp, ctx, p_ids, pre_ids, li, scaling, max_len)
    lp_post = _span_logprob(base, tok, hyp, ctx, p_ids, post_ids, li, scaling, max_len)
    if lp_pre is None or lp_post is None:
        return None
    return lp_post - lp_pre


def main() -> int:
    ap = argparse.ArgumentParser()
    # c3 (Phase-1 best / Phase-2 retention baseline): tau=-0.7 lp2 lg1,
    # MLflow run fe72f9ddd69c, sha256 53e24af2…
    ap.add_argument("--ckpt", default="/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt")
    # Recovered external_codereview val split (cross-domain gate corpus).
    # Durable: s3://elixirtrials-949678234935-us-east-1-artifacts/training-data/
    #   github-pairs/splits/external_codereview.val.clean.jsonl (sha256 7e3692df…).
    # Local copy below is bit-identical to that S3 artifact.
    ap.add_argument(
        "--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl"
    )
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument(
        "--scaling",
        type=float,
        default=None,
        help="adapter scaling for matched/mismatch passes; default = lora_alpha "
        "(the shared contract). zero/base pass always uses 0.0.",
    )
    ap.add_argument("--tail-lines", type=int, default=5)
    ap.add_argument("--max-seq-length", type=int, default=768)
    # The adapter is conditioned on the base model's activations, so the eval
    # base MUST match the one the ckpt was trained/evaluated against. c3 was
    # produced with Qwen3-4B-Instruct-2507 in bf16 (orchestrator: load_in_4bit
    # false, _specificity_probe with no --load-4bit). 4-bit is opt-in only.
    ap.add_argument("--model-id", default=load_rune_config().model_id)
    ap.add_argument(
        "--load-4bit",
        action="store_true",
        help="load base in 4-bit nf4 (default bf16, matching the c3 recipe).",
    )
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    load_kw = dict(
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    )
    if a.load_4bit:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    print(
        f"base={a.model_id}  dtype={'4bit-nf4' if a.load_4bit else 'bf16'}", flush=True
    )
    base = AutoModelForCausalLM.from_pretrained(a.model_id, **load_kw).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=a.ckpt), device="cuda")
    hyp.eval()
    li = list(hyp.config.layer_indices)
    # Resolve scaling to the shared contract (lora_alpha) unless overridden on the CLI.
    if a.scaling is None:
        a.scaling = effective_scaling(hyp)

    rows = []
    with open(a.val) as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            m = _map_record(raw)
            if not m:
                continue
            rows.append(
                {
                    "ctx": m["context"],
                    "answer": m["answer"],
                    "pre_code": str(raw.get("pre_code", "")),
                    "post_code": str(raw.get("post_code", "")),
                    "feedback": extract_review_feedback(m["context"]) or "",
                }
            )
            if len(rows) >= a.n:
                break

    n = len(rows)
    print(f"ckpt={a.ckpt}  n={n}  scaling={a.scaling}  tail_lines={a.tail_lines}")
    # precompute prompt/target token ids per target
    agg = {t: {"m": [], "x": [], "z": []} for t in ("goal", "diff", "tail", "avoid")}
    fb_prompt = tok("## Review Feedback\n", add_special_tokens=False)["input_ids"]
    cc_prompt = tok("## Current Code\n", add_special_tokens=False)["input_ids"]
    for i, r in enumerate(rows):
        other = rows[(i + 1) % n]["ctx"]
        # GOAL: recover the review feedback under "## Review Feedback\n"
        if r["feedback"]:
            ft = tok(r["feedback"], add_special_tokens=False)["input_ids"][:128]
            for key, ctx, sc in (
                ("m", r["ctx"], a.scaling),
                ("x", other, a.scaling),
                ("z", r["ctx"], 0.0),
            ):
                v = _span_logprob(
                    base, tok, hyp, ctx, fb_prompt, ft, li, sc, a.max_seq_length
                )
                if v is not None:
                    agg["goal"][key].append(v)
        # DIFF: recover the edit (edit-local tokens of the revision)
        for key, ctx, sc in (
            ("m", r["ctx"], a.scaling),
            ("x", other, a.scaling),
            ("z", r["ctx"], 0.0),
        ):
            v = _diff_logprob(
                base,
                tok,
                hyp,
                ctx,
                r["answer"],
                r["pre_code"],
                li,
                sc,
                a.max_seq_length,
            )
            if v is not None:
                agg["diff"][key].append(v)
        # TAIL: last N lines of current code (drives next step) given the earlier code
        lines = r["pre_code"].splitlines(keepends=True)
        if len(lines) > a.tail_lines + 2:
            prefix = "".join(lines[: -a.tail_lines])
            tail = "".join(lines[-a.tail_lines :])
            prompt_ids = (
                cc_prompt
                + tok(prefix, add_special_tokens=False)["input_ids"][
                    -(a.max_seq_length - 160) :
                ]
            )
            tt = tok(tail, add_special_tokens=False)["input_ids"][:128]
            for key, ctx, sc in (
                ("m", r["ctx"], a.scaling),
                ("x", other, a.scaling),
                ("z", r["ctx"], 0.0),
            ):
                v = _span_logprob(
                    base, tok, hyp, ctx, prompt_ids, tt, li, sc, a.max_seq_length
                )
                if v is not None:
                    agg["tail"][key].append(v)
        # AVOID-FAILED: prefer accepted (post) over rejected (pre) form at the edit hunk
        if r["pre_code"] and r["post_code"]:
            for key, ctx, sc in (
                ("m", r["ctx"], a.scaling),
                ("x", other, a.scaling),
                ("z", r["ctx"], 0.0),
            ):
                v = _avoid_failed_margin(
                    base,
                    tok,
                    hyp,
                    ctx,
                    r["pre_code"],
                    r["post_code"],
                    li,
                    sc,
                    a.max_seq_length,
                )
                if v is not None:
                    agg["avoid"][key].append(v)

    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    print(f"\n{'target':6} (n)   matched   mismatch  zero    | m-mismatch  m-zero")
    for t in ("goal", "diff", "tail", "avoid"):
        mm, xx, zz = mean(agg[t]["m"]), mean(agg[t]["x"]), mean(agg[t]["z"])
        print(
            f"{t:6} ({len(agg[t]['m']):2})  {mm:8.4f}  {xx:8.4f}  {zz:8.4f} | {mm - xx:+.5f}   {mm - zz:+.4f}"
        )
    print(
        "\nREAD: the bet needs m-mismatch>0 (episode-specific) AND m-zero>0 (beats no-context). "
        "tail = semi-Markov 'drives next step'; avoid = logp(accepted)-logp(rejected) at the "
        "edit hunk, m-mismatch>0 = the adapter episode-specifically avoids the failed approach."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
