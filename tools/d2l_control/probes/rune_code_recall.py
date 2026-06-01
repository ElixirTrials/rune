"""Issue #52 — can Sakana's Doc2LoRA recall a CODE snippet / answer questions about it?

Internalize a short code document into the adapter, then (with the code NOT in the prompt)
test recall two ways:
  (1) logprob scorecard: mean gold logprob of an answer span under matched / mismatch / zero;
  (2) generative QA: greedy-generate the answer and check it.
Five fact types per snippet: function name, a magic constant, a return value, a verbatim
continuation, and a free-form "what does it do". Run from this dir's venv (flash, patches inert).
"""
import sys

import torch

from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

import os
CKPT = os.environ.get("D2L_CKPT", "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin")

# Two distinct code snippets with unguessable facts (so recall != prior).
SNIPPETS = [
    {
        "doc": (
            "```python\n"
            "RETRY_BUDGET = 7\n\n"
            "def quarkle_resync(payload, attempts):\n"
            "    # reconcile the ledger against the upstream shard\n"
            "    if attempts > RETRY_BUDGET:\n"
            "        return 'ABANDONED'\n"
            "    checksum = (payload * 31 + 17) % 9973\n"
            "    return checksum\n"
            "```\n"
        ),
        "facts": {
            "fn_name": ("What is the name of the function defined in the code?", "quarkle_resync"),
            "constant": ("What integer does RETRY_BUDGET equal in the code?", "7"),
            "return_str": ("What string does the function return when attempts exceed the budget?", "ABANDONED"),
            "modulus": ("In the checksum formula, what number is used as the modulus?", "9973"),
        },
    },
    {
        "doc": (
            "```python\n"
            "MAX_DEPTH = 42\n\n"
            "def frobnicate_tree(node, depth):\n"
            "    # walk the spanning tree breadth-first\n"
            "    if depth >= MAX_DEPTH:\n"
            "        return 'TRUNCATED'\n"
            "    weight = (node ^ 53) + 101\n"
            "    return weight\n"
            "```\n"
        ),
        "facts": {
            "fn_name": ("What is the name of the function defined in the code?", "frobnicate_tree"),
            "constant": ("What integer does MAX_DEPTH equal in the code?", "42"),
            "return_str": ("What string does the function return when depth reaches the max?", "TRUNCATED"),
            "modulus": ("In the weight formula, what number is added at the end?", "101"),
        },
    },
]


def ctx_ids_for(model, ctx_tok, doc):
    ids = tokenize_ctx_text(dict(context=[doc]), ctx_tok)["ctx_ids"]
    return torch.tensor(ids, device=model.device)


def logits_with_ctx(model, full_ids, ctx_ids):
    model.reset()
    model.patch_lora_forward()
    out = model(
        ctx_ids=ctx_ids,
        ctx_attn_mask=torch.ones_like(ctx_ids),
        n_ctx_chunks=torch.tensor([1], device=model.device),
        n_queries=torch.tensor([1], device=model.device),
        input_ids=full_ids,
    )
    return out.logits[0]


def logits_zero(model, full_ids):
    model.reset()
    return model.base_model(input_ids=full_ids).logits[0]


def span_lp(logits, ids, start, length):
    lp = torch.log_softmax(logits.float(), dim=-1)
    return sum(float(lp[t - 1, ids[0, t]]) for t in range(start, start + length)) / length


def build_full(tok, model, query, answer):
    chat = [{"role": "user", "content": query}]
    p = tok.apply_chat_template(
        chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    a = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    return p, torch.cat([p, a], dim=1), p.shape[1], a.shape[1]


def gen_answer(model, tok, doc, query, max_new=24):
    chat = [{"role": "user", "content": query}]
    p = tok.apply_chat_template(
        chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    model.reset()
    model.internalize(doc)
    out = model.generate(input_ids=p, max_new_tokens=max_new)
    model.reset()
    return tok.decode(out[0][p.shape[1]:], skip_special_tokens=True).strip()


def main():
    print("loading...", flush=True)
    sd = torch.load(CKPT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        sd, train=False, use_sequence_packing=False, use_flash_attn=False
    )
    model.reset()
    tok = get_tokenizer(model.base_model.name_or_path)
    ctx_tok = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    ctxs = [ctx_ids_for(model, ctx_tok, s["doc"]) for s in SNIPPETS]
    rows = []
    for i, snip in enumerate(SNIPPETS):
        mismatch_ctx = ctxs[(i + 1) % len(SNIPPETS)]  # the OTHER snippet's adapter
        for fkey, (query, answer) in snip["facts"].items():
            p, full, start, length = build_full(tok, model, query, answer)
            lp_m = span_lp(logits_with_ctx(model, full, ctxs[i]), full, start, length)
            lp_x = span_lp(logits_with_ctx(model, full, mismatch_ctx), full, start, length)
            lp_z = span_lp(logits_zero(model, full), full, start, length)
            gen = gen_answer(model, tok, snip["doc"], query)
            hit = answer.lower() in gen.lower()
            rows.append((i, fkey, answer, lp_m, lp_x, lp_z, lp_m - lp_x, lp_m - lp_z, gen, hit))
            print(f"[snip{i}:{fkey}] ans={answer!r}  m={lp_m:.3f} mis={lp_x:.3f} zero={lp_z:.3f} "
                  f"| m-mis={lp_m-lp_x:+.3f} m-zero={lp_m-lp_z:+.3f} | gen={gen!r} hit={hit}", flush=True)

    n = len(rows)
    mm = sum(r[6] for r in rows) / n
    mz = sum(r[7] for r in rows) / n
    acc = sum(r[9] for r in rows) / n
    spec = sum(1 for r in rows if r[6] > 0) / n
    print(f"\n=== CODE RECALL SUMMARY (n={n}) ===", flush=True)
    print(f"mean m-mismatch={mm:+.3f}  mean m-zero={mz:+.3f}  "
          f"gen_accuracy={acc:.2f}  frac(m-mismatch>0)={spec:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
