"""Issue #52 — eval-only: which facets bind the TRAJECTORY FACT vs just echo the code?

For goal and diff, measure zero-shot m-mismatch against TWO negatives:
  generic       = a different episode's adapter (easy; can be won by code/metadata echo)
  feedback-swap = SAME code/file, DIFFERENT review feedback (hard; only winnable by binding
                  the trajectory request)
Conjecture test: feedback-derived facts (goal) should hold under feedback-swap; the diff
(code output) should collapse -> diff is code-echo, a bad memory target. No training (no OOM).
"""
import os
import sys

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools"); sys.path.insert(0, f"{RUNE}/tools/d2l_control")
import scoring_core  # noqa: E402
from episodes import build_rune_episodes, extract_review_feedback  # noqa: E402
from ctx_to_lora.data.processing import tokenize_ctx_text  # noqa: E402
from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

CKPT = os.environ.get("D2L_CKPT", "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin")
VAL = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"
MAX_ANS = 48


def cx(model, ct, doc):
    return torch.tensor(tokenize_ctx_text(dict(context=[doc]), ct)["ctx_ids"], device=model.device)


def lg(model, full, ctx):
    model.reset(); model.patch_lora_forward()
    with torch.no_grad():
        return model(ctx_ids=ctx, ctx_attn_mask=torch.ones_like(ctx),
                     n_ctx_chunks=torch.tensor([1], device=model.device),
                     n_queries=torch.tensor([1], device=model.device), input_ids=full).logits[0]


def full_ids(tok, model, q, a):
    p = tok.apply_chat_template([{"role": "user", "content": q}], add_special_tokens=False,
                                add_generation_prompt=True, return_tensors="pt").to(model.device)
    aa = torch.tensor([tok(a, add_special_tokens=False).input_ids[:MAX_ANS]], device=model.device)
    return torch.cat([p, aa], dim=1), p.shape[1], aa.shape[1]


def main():
    sd = torch.load(CKPT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(sd, train=False, use_sequence_packing=False, use_flash_attn=False)
    model.reset()
    tok = get_tokenizer(model.base_model.name_or_path)
    ct = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    eps = build_rune_episodes(VAL, n=12)

    res = {}
    for tgt in ("goal", "diff"):
        gen, hard = [], []
        for i, e in enumerate(eps):
            q = e.queries[tgt]
            full, s, n = full_ids(tok, model, q["query"], q["answer"])
            if n < 1:
                continue
            other = eps[(i + 1) % len(eps)]
            m = scoring_core.mean_gold_logprob(lg(model, full, cx(model, ct, e.doc)), full[0], s, n)
            xg = scoring_core.mean_gold_logprob(lg(model, full, cx(model, ct, other.doc)), full[0], s, n)
            gen.append(m - xg)
            hn = extract_review_feedback(e.doc); on = extract_review_feedback(other.doc)
            if hn and on and hn in e.doc:
                sw = e.doc.replace(hn, on)  # same code, swapped feedback
                xh = scoring_core.mean_gold_logprob(lg(model, full, cx(model, ct, sw)), full[0], s, n)
                hard.append(m - xh)
        res[tgt] = (sum(gen) / len(gen), sum(hard) / len(hard) if hard else float("nan"), len(hard))
        print(f"{tgt:5s}  m-mismatch(generic)={res[tgt][0]:+.3f}  m-mismatch(feedback-swap)={res[tgt][1]:+.3f}  (n_hard={res[tgt][2]})", flush=True)

    print("\n=== CONJECTURE READ ===", flush=True)
    print(f"  goal holds under feedback-swap: {res['goal'][1]:+.3f}  | diff under feedback-swap: {res['diff'][1]:+.3f}", flush=True)
    print("  If goal >> diff under feedback-swap: goal binds the trajectory fact; diff is code-echo (bad memory target).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
