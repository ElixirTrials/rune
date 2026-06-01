"""Issue #52 — CONTINUATION facet: does Sakana's adapter recall the specific code body
('tail'/'drives the next step', the sharpest Rune #49 failure: m-zero -0.38)?

Internalize the full snippet; prompt with ONLY the signature/prefix (body NOT in prompt);
score the continuation (body) under matched / mismatch / zero, plus greedy generation.
Run from this dir's venv. D2L_CKPT env selects the checkpoint (default gemma_demo).
"""
import os
import sys

import torch

from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

CKPT = os.environ.get("D2L_CKPT", "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin")

# (full doc, prefix shown in prompt, continuation = the body to recall)
CASES = [
    (
        "```python\nRETRY_BUDGET = 7\n\ndef quarkle_resync(payload, attempts):\n"
        "    if attempts > RETRY_BUDGET:\n        return 'ABANDONED'\n"
        "    checksum = (payload * 31 + 17) % 9973\n    return checksum\n```\n",
        "RETRY_BUDGET = 7\n\ndef quarkle_resync(payload, attempts):\n",
        "    if attempts > RETRY_BUDGET:\n        return 'ABANDONED'\n"
        "    checksum = (payload * 31 + 17) % 9973\n    return checksum\n",
    ),
    (
        "```python\nMAX_DEPTH = 42\n\ndef frobnicate_tree(node, depth):\n"
        "    if depth >= MAX_DEPTH:\n        return 'TRUNCATED'\n"
        "    weight = (node ^ 53) + 101\n    return weight\n```\n",
        "MAX_DEPTH = 42\n\ndef frobnicate_tree(node, depth):\n",
        "    if depth >= MAX_DEPTH:\n        return 'TRUNCATED'\n"
        "    weight = (node ^ 53) + 101\n    return weight\n",
    ),
]


def ctx_ids_for(model, ctx_tok, doc):
    return torch.tensor(tokenize_ctx_text(dict(context=[doc]), ctx_tok)["ctx_ids"], device=model.device)


def logits_with_ctx(model, full_ids, ctx_ids):
    model.reset()
    model.patch_lora_forward()
    out = model(ctx_ids=ctx_ids, ctx_attn_mask=torch.ones_like(ctx_ids),
                n_ctx_chunks=torch.tensor([1], device=model.device),
                n_queries=torch.tensor([1], device=model.device), input_ids=full_ids)
    return out.logits[0]


def logits_zero(model, full_ids):
    model.reset()
    return model.base_model(input_ids=full_ids).logits[0]


def span_lp(logits, ids, start, length):
    lp = torch.log_softmax(logits.float(), dim=-1)
    return sum(float(lp[t - 1, ids[0, t]]) for t in range(start, start + length)) / length


def main():
    sd = torch.load(CKPT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(sd, train=False, use_sequence_packing=False, use_flash_attn=False)
    model.reset()
    tok = get_tokenizer(model.base_model.name_or_path)
    ctx_tok = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    ctxs = [ctx_ids_for(model, ctx_tok, doc) for doc, _, _ in CASES]
    mm, mz = [], []
    for i, (doc, prefix, cont) in enumerate(CASES):
        chat = [{"role": "user", "content": "Complete this code:\n```python\n" + prefix}]
        p = tok.apply_chat_template(chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt").to(model.device)
        c = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        full = torch.cat([p, c], dim=1)
        s, n = p.shape[1], c.shape[1]
        lp_m = span_lp(logits_with_ctx(model, full, ctxs[i]), full, s, n)
        lp_x = span_lp(logits_with_ctx(model, full, ctxs[(i + 1) % len(CASES)]), full, s, n)
        lp_z = span_lp(logits_zero(model, full), full, s, n)
        model.reset(); model.internalize(doc)
        gen = tok.decode(model.generate(input_ids=p, max_new_tokens=48)[0][p.shape[1]:], skip_special_tokens=True)
        model.reset()
        key = "9973" if i == 0 else "101"
        mm.append(lp_m - lp_x); mz.append(lp_m - lp_z)
        print(f"[case{i}] cont_len={n} m={lp_m:.3f} mis={lp_x:.3f} zero={lp_z:.3f} "
              f"| m-mis={lp_m-lp_x:+.3f} m-zero={lp_m-lp_z:+.3f} | gen_has_{key}={key in gen}", flush=True)
        print(f"        gen: {gen!r}", flush=True)
    print(f"\n=== CONTINUATION SUMMARY === mean m-mismatch={sum(mm)/len(mm):+.3f} mean m-zero={sum(mz)/len(mz):+.3f} "
          f"(Rune #49 tail m-zero was -0.38)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
