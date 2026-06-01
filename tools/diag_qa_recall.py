"""Can we get the EPISODE back out of the adapter? Q&A / recall-out test (#49).

The adapter-as-episodic-memory bet (Sakana/doc2lora style): after embedding an episode
into W(ctx), base+W(ctx) should let us RECOVER facts of that episode that base alone
cannot, and that base+W(other_episode) cannot. Prior probes tested reproduction of the
gold EDIT (teacher-forced). This tests recall of an episode FACT not in the training
output: the REVIEW FEEDBACK (the trajectory-specific request) and the FILE path.

For each row we score the fact span under a NEUTRAL lead-in prompt (the episode is NOT
in the prompt — only the adapter carries it):
  matched   = adapter generated from THIS row's context
  mismatched = adapter from a DIFFERENT row's context
  zero      = base, no adapter
matched >> mismatched/zero  => the episode fact is recoverable from the adapter.
matched ~ mismatched ~ zero => no recoverable episodic content.

Also emits a few FREE GENERATIONS (greedy) from base+matched-adapter on the lead-in, to
show qualitatively what "comes out of the adapter" vs the true fact.
No training; base loaded once. Run under tools/run_guarded.sh.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import torch

from rune.model.hypernetwork import HypernetworkConfig, load_hypernetwork
from rune.training.contrastive import extract_review_feedback
from rune.training.hypernet_distill import (
    _functional_lora,
    _generate_lora_dict,
    _map_record,
)

_FILE_RE = re.compile(r"file:\s*([^)\n]+)")


def _fact_logprob(base, tok, hyp, ctx, prompt, fact, li, scaling, max_len):
    """Mean gold logprob of `fact` tokens given `prompt`, under ctx's adapter (or base)."""
    device = next(base.parameters()).device
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(fact, add_special_tokens=False)["input_ids"][:128]
    if len(f_ids) < 1:
        return None
    ids = torch.tensor([p_ids + f_ids], device=device)
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
        # positions predicting the fact tokens: (len(p)-1 .. end-1)
        start = len(p_ids)
        tot = 0.0
        for t in range(start, start + len(f_ids)):
            tot += float(lp[t - 1, ids[0, t]])
    return tot / len(f_ids)


def _generate(base, tok, hyp, ctx, prompt, li, scaling, max_len, new=40):
    device = next(base.parameters()).device
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    n_qs = torch.tensor([1], device=device)
    ld = _generate_lora_dict(hyp, ctx, base, tok, li, max_len)
    with torch.no_grad(), _functional_lora(base, li, ld, scaling, n_qs):
        cur = list(ids)
        for _ in range(new):
            out = base(torch.tensor([cur], device=device), use_cache=False).logits[
                0, -1
            ]
            cur.append(int(out.argmax()))
    del ld
    return tok.decode(cur[len(ids) :])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/rune-ck-final/checkpoint_step600.pt")
    ap.add_argument(
        "--val", default="/tmp/rune-corpus/external_codereview.val.clean.jsonl"
    )
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--scaling", type=float, default=0.5)
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
            raw = json.loads(line)
            m = _map_record(raw)
            if not m:
                continue
            fb = extract_review_feedback(m["context"]) or ""
            fm = _FILE_RE.search(m["context"])
            rows.append(
                {
                    "ctx": m["context"],
                    "feedback": fb,
                    "file": fm.group(1).strip() if fm else "",
                }
            )
            if len(rows) >= a.n:
                break

    n = len(rows)
    print(f"ckpt={a.ckpt}  n={n}  scaling={a.scaling}")
    for fact_key, prompt in [("feedback", "## Review Feedback\n"), ("file", "file: ")]:
        m_, x_, z_ = [], [], []
        for i, r in enumerate(rows):
            fact = r[fact_key]
            if not fact:
                continue
            other = rows[(i + 1) % n]["ctx"]
            lm = _fact_logprob(
                base, tok, hyp, r["ctx"], prompt, fact, li, a.scaling, a.max_seq_length
            )
            lx = _fact_logprob(
                base, tok, hyp, other, prompt, fact, li, a.scaling, a.max_seq_length
            )
            lz = _fact_logprob(
                base, tok, hyp, r["ctx"], prompt, fact, li, 0.0, a.max_seq_length
            )
            if None in (lm, lx, lz):
                continue
            m_.append(lm)
            x_.append(lx)
            z_.append(lz)
        mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
        mm, xx, zz = mean(m_), mean(x_), mean(z_)
        print(
            f"\nRECALL[{fact_key}] (n={len(m_)}): matched={mm:.4f} mismatch={xx:.4f} zero={zz:.4f} "
            f"| m-mismatch={mm - xx:+.4f} m-zero={mm - zz:+.4f}"
        )

    print(
        "\n=== free generation from base+matched-adapter on '## Review Feedback\\n' ==="
    )
    for r in rows[:3]:
        gen = _generate(
            base,
            tok,
            hyp,
            r["ctx"],
            "## Review Feedback\n",
            li,
            a.scaling,
            a.max_seq_length,
        )
        print(f"\nTRUE feedback : {r['feedback'][:160]!r}")
        print(f"GENERATED     : {gen[:160]!r}")
    print(
        "\nREAD: RECALL m-mismatch >0 => episode fact recoverable from adapter; "
        "~0 => no recoverable episodic content (adapter is context-invariant)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
