"""Issue #52 — model-load + recall smoke for the Doc2LoRA positive control.

Decisive first integration step (spec §5 step 1): confirm we can (a) load the Sakana
gemma_demo checkpoint, (b) get per-token LOGITS with the adapter active via forward()
(not just generate()), and (c) see matched > zero on a needle. Run from this dir's venv.
"""
import sys

import torch

from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

CKPT = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"


def ctx_ids_for(model, ctx_tok, doc):
    ids = tokenize_ctx_text(dict(context=[doc]), ctx_tok)["ctx_ids"]
    return torch.tensor(ids, device=model.device)


def logits_with_ctx(model, full_ids, ctx_ids):
    """Teacher-forced logits over full_ids with the adapter generated from ctx_ids."""
    model.reset()
    model.patch_lora_forward()  # reinstall lora_forward; forward() only binds A/B on top
    attn = torch.ones_like(ctx_ids)
    out = model(
        ctx_ids=ctx_ids,
        ctx_attn_mask=attn,
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
    tot = sum(float(lp[t - 1, ids[0, t]]) for t in range(start, start + length))
    return tot / length


def main():
    print("loading checkpoint...", flush=True)
    sd = torch.load(CKPT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        sd, train=False, use_sequence_packing=False, use_flash_attn=False
    )
    model.reset()
    base_name = model.base_model.name_or_path
    print("base model:", base_name, "| device:", model.device, flush=True)
    tok = get_tokenizer(base_name)
    ctx_tok = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    print("use_base_input_as_ctx:", getattr(model, "use_base_input_as_ctx", None), flush=True)

    # Two needle docs (magic-number style) and a shared query.
    doc_a = (
        "The grass is green. The sky is blue. The sun is yellow. Here we go. "
        "The special magic number is 4417. There and back again. The minutes were filed."
    )
    doc_b = (
        "The grass is green. The sky is blue. The sun is yellow. Here we go. "
        "The special magic number is 9023. There and back again. The minutes were filed."
    )
    query = "What is the special magic number?"
    answer = "4417"  # the needle in doc_a

    chat = [{"role": "user", "content": query}]
    prompt_ids = tok.apply_chat_template(
        chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    ans_ids = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    full_ids = torch.cat([prompt_ids, ans_ids], dim=1)
    start, length = prompt_ids.shape[1], ans_ids.shape[1]
    print(f"prompt_len={start} ans_len={length} ans_ids={ans_ids.tolist()}", flush=True)

    ctx_a = ctx_ids_for(model, ctx_tok, doc_a)
    ctx_b = ctx_ids_for(model, ctx_tok, doc_b)

    lp_match = span_lp(logits_with_ctx(model, full_ids, ctx_a), full_ids, start, length)
    lp_mis = span_lp(logits_with_ctx(model, full_ids, ctx_b), full_ids, start, length)
    lp_zero = span_lp(logits_zero(model, full_ids), full_ids, start, length)
    print(f"\nNEEDLE '4417' logprob:  matched(doc_a)={lp_match:.4f}  "
          f"mismatch(doc_b)={lp_mis:.4f}  zero={lp_zero:.4f}", flush=True)
    print(f"  m-mismatch={lp_match - lp_mis:+.4f}   m-zero={lp_match - lp_zero:+.4f}", flush=True)

    # Generation recall (their canonical path).
    model.reset()
    model.internalize(doc_a)
    gen = model.generate(input_ids=prompt_ids, max_new_tokens=16)
    gen_txt = tok.decode(gen[0][prompt_ids.shape[1]:], skip_special_tokens=True)
    model.reset()
    print(f"\nGEN (matched doc_a): {gen_txt!r}  -> contains '4417': {'4417' in gen_txt}", flush=True)

    ok = (lp_match > lp_mis) and (lp_match > lp_zero)
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'}: matched beats mismatch and zero on the needle", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
