"""Feedback-binding success metric (the training go/no-go, advisor) — T0 paired harness.

The training loop OPTIMIZES the right objective (contrastive hinge on edit-local
tokens with a feedback-swapped hard negative) but SELECTS on val_diff_agreement,
which is matched-vs-base = the discipline confound (goal 3). This eval is the
discriminating metric: on held-out episodes, does the adapter BIND the feedback?

  matched  : adapter from the episode's own context.
  swap     : adapter from make_hard_negative(context, other_feedback) — SAME Task
             + Current Code, feedback CONTENT replaced by a different row's feedback.
  zero     : base, no adapter (shared across arms).
Score = mean gold logprob over EDIT-LOCAL answer tokens (edit_local_mask), the
exact span the contrastive loss uses. _student_logits scores the answer ALONE with
the adapter applied (Doc2LoRA: context internalized, not in prompt).

Margins: matched - swap  = feedback-binding (the success metric); matched - zero =
discipline (secondary).

T0 (predeclared spec, 2026-06-02): pass TWO checkpoints (--ckpt = warm-start arm1,
--ckpt2 = trained arm2) so both are scored in ONE process on BYTE-IDENTICAL rows under
ONE fixed --max-seq-length (threaded to BOTH the adapter-conditioning and the scored
span). This makes the paired per-episode delta (arm2 - arm1) valid — the historical
+0.0185 (2048-path) vs +0.0687 (768-path) comparison is FORBIDDEN (uncontrolled knob).
Per-episode rows are dumped to --out for paired bootstrap CI + sign test (see
tools/_feedback_swap_stats.py); this script computes no aggregate inferential stat.

  uv run python tools/_feedback_swap_eval.py --ckpt <warm> --ckpt2 <trained> \\
      --max-seq-length 768 --n 60 --out /tmp/t0_dump.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
CKPT = (
    f"{RUNE}/third_party/doc-to-lora/trained_d2l/"
    "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
BASE = "Qwen/Qwen3-4B-Instruct-2507"
CORPUS = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT, help="arm1 (warm-start baseline)")
    ap.add_argument("--ckpt2", type=str, default=None, help="arm2 (trained); omit for single-arm")
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--corpus", type=str, default=CORPUS)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-seq-length", type=int, default=768,
                    help="ONE value threaded to BOTH conditioning + scored span (training regime)")
    ap.add_argument("--bf16", action="store_true", help="bf16 base (default 4bit = train)")
    ap.add_argument("--out", type=str, default=None, help="per-episode JSONL dump path")
    a = ap.parse_args()

    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from rune.model.adapter_contract import assemble_adapter, effective_scaling  # noqa: PLC0415,E501
    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        load_hypernetwork,
    )
    from rune.training.contrastive import (  # noqa: PLC0415
        edit_local_mask,
        extract_review_feedback,
        make_hard_negative,
    )
    from rune.training.hypernet_distill import (  # noqa: PLC0415
        _generate_lora_dict,
        _map_record,
        _prepare_ids,
        _student_logits,
    )

    load_kw = dict(dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                   device_map={"": "cuda"})
    if not a.bf16:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    print(f"[load] base={a.model_id} ({'bf16' if a.bf16 else '4bit'}) "
          f"max_seq_length={a.max_seq_length}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(a.model_id, **load_kw).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    device = next(base.parameters()).device
    n_chunks = torch.tensor([1], device=device)

    def load_arm(ckpt: str, name: str) -> dict:
        print(f"[load] arm {name} ckpt={ckpt}", flush=True)
        hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=ckpt), device="cuda")
        hyp.eval()
        return {"name": name, "hyp": hyp, "eff": effective_scaling(hyp),
                "li": [int(x) for x in hyp.config.layer_indices]}

    arms = [load_arm(a.ckpt, "arm1")]
    if a.ckpt2:
        arms.append(load_arm(a.ckpt2, "arm2"))

    # map records ONCE (deterministic, checkpoint-independent) — both arms see identical rows
    recs = []
    with open(a.corpus) as f:
        for line in f:
            rec = json.loads(line)
            m = _map_record(rec)
            if m is None:
                continue
            m["pre_code"] = str(rec.get("pre_code", ""))
            m["feedback"] = extract_review_feedback(m["context"])
            m["task_id"] = rec.get("task_id")
            if m["feedback"]:
                recs.append(m)
            if len(recs) >= a.n:
                break
    pool = [m["feedback"] for m in recs]
    print(f"[data] {len(recs)} episodes with feedback", flush=True)

    def edit_local_lp(logits, ans_ids, em) -> float:
        gold = torch.tensor(ans_ids[1:], device=logits.device)
        lp = (
            torch.log_softmax(logits[:-1].float(), dim=-1)
            .gather(-1, gold.unsqueeze(-1))
            .squeeze(-1)
        )
        emt = torch.tensor(em[1:], device=logits.device, dtype=torch.bool)
        if int(emt.sum()) == 0:
            return float("nan")
        return float(lp[emt].mean())

    def logits_for(arm, ctx, ans_ids):
        ld = _generate_lora_dict(arm["hyp"], ctx, base, tok, arm["li"], a.max_seq_length)
        asm = assemble_adapter(arm["hyp"], ld, n_chunks)
        return _student_logits(base, tok, ans_ids, asm, arm["li"], arm["eff"])

    # ---- Pass 1: checkpoint-INDEPENDENT eligibility (depends only on tok+max_seq_length+pre_code) ----
    prepared = []  # list of dicts for eligible rows
    for i, m in enumerate(recs):
        _full, ans_ids = _prepare_ids(tok, m["context"], m["answer"], a.max_seq_length)
        em = edit_local_mask(tok, m["pre_code"], ans_ids)
        ctx_hash = hashlib.sha1(
            (m["context"] + "\x00" + m["answer"] + "\x00" + m["pre_code"]).encode()
        ).hexdigest()
        skip = None
        if len(ans_ids) < 2:
            skip = "ans<2"
        elif sum(em[1:]) == 0:
            skip = "no_edit"
        prepared.append({"row_idx": i, "task_id": m.get("task_id"), "m": m,
                         "ans_ids": ans_ids, "em": em, "ctx_hash": ctx_hash,
                         "n_ans_tok": len(ans_ids), "n_edit_tok": int(sum(em[1:])),
                         "neg_idx": (i + 1) % len(pool), "skip": skip})

    # ---- Pass 2: per-arm scoring (zero shared); NaN pairing -> shared denominator ----
    dump = []
    for p in prepared:
        row = {"row_idx": p["row_idx"], "task_id": p["task_id"],
               "n_ans_tok": p["n_ans_tok"], "n_edit_tok": p["n_edit_tok"],
               "neg_idx": p["neg_idx"], "ctx_hash": p["ctx_hash"],
               "eligible": p["skip"] is None, "skip_reason": p["skip"]}
        if p["skip"] is not None:
            dump.append(row)
            continue
        m, ans_ids, em = p["m"], p["ans_ids"], p["em"]
        neg_fb = pool[p["neg_idx"]]
        neg_ctx = make_hard_negative(m["context"], other_feedback=neg_fb)
        with torch.no_grad():
            ans_only = torch.tensor([ans_ids], device=device)
            lp_z = edit_local_lp(base(ans_only, use_cache=False).logits[0], ans_ids, em)
            for arm in arms:
                lp_m = edit_local_lp(logits_for(arm, m["context"], ans_ids), ans_ids, em)
                lp_n = edit_local_lp(logits_for(arm, neg_ctx, ans_ids), ans_ids, em)
                row[arm["name"]] = {
                    "lp_m": lp_m, "lp_n": lp_n, "lp_z": lp_z,
                    "matched_swap": lp_m - lp_n, "matched_zero": lp_m - lp_z,
                }
        # NaN pairing: any arm NaN (or zero NaN) -> drop from BOTH means
        def arm_nan(name):
            r = row[name]
            return any(x != x for x in (r["lp_m"], r["lp_n"], r["lp_z"]))
        nan_arms = [arm["name"] for arm in arms if arm_nan(arm["name"])]
        if nan_arms:
            row["eligible"] = False
            row["skip_reason"] = "nan_" + "+".join(nan_arms)
        dump.append(row)

    # ---- Aggregate (descriptive only; inference in _feedback_swap_stats.py from --out) ----
    scored = [r for r in dump if r["eligible"]]
    print(f"\n[denominator] eligible after NaN pairing: {len(scored)} "
          f"(of {len(prepared)} prepared, {sum(1 for p in prepared if p['skip'])} pre-skipped)", flush=True)
    for arm in arms:
        ms = [r[arm["name"]]["matched_swap"] for r in scored]
        mz = [r[arm["name"]]["matched_zero"] for r in scored]
        n = len(ms)
        if n == 0:
            print(f"  {arm['name']}: no eligible rows", flush=True)
            continue
        mm = sum(ms) / n
        mzz = sum(mz) / n
        fmm = sum(1 for x in ms if x > 0) / n
        print(f"  {arm['name']} ({'ckpt' if arm['name']=='arm1' else 'ckpt2'}): "
              f"matched-SWAP={mm:+.4f} frac(>0)={fmm:.2f}  matched-zero={mzz:+.4f}  n={n}",
              flush=True)
    # n_arm1 == n_arm2 assertion: identical denominator by construction (shared `scored` set)
    if len(arms) == 2:
        n1 = sum(1 for r in scored if "arm1" in r)
        n2 = sum(1 for r in scored if "arm2" in r)
        assert n1 == n2, f"denominator mismatch n_arm1={n1} n_arm2={n2}"
        print(f"[assert] n_arm1==n_arm2=={n1}  (shared denominator OK)", flush=True)

    if a.out:
        with open(a.out, "w") as f:
            for r in dump:
                f.write(json.dumps(r) + "\n")
        print(f"[dump] {len(dump)} rows -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
