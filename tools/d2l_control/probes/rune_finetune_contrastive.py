"""Issue #52 — CONTRASTIVE light-finetune (the fix-test for diff specificity).

Plain answer-CE raised diff m-zero but DROPPED diff m-mismatch (#49 generic-boost). Here the
objective is specificity-aware: CE(answer) + lambda * hinge[margin - (lp_matched - lp_hardneg)]
on the diff answer, where the HARD NEGATIVE is a CONSTRUCTED feedback-swap (SAME code/file/format,
DIFFERENT review feedback) — so the only way to win the margin is to bind the trajectory fact,
not emit generic diff tokens (reviewer: facet-paired constructed negatives, not generic episodes).
Eval diff m-mismatch vs BOTH a generic other-episode AND the feedback-swap hard negative; +
retention (NIAH/code) + ||Δhypernet|| ("how light"). Run from this dir's venv under run_guarded.
"""
import os
import sys

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
sys.path.insert(0, f"{RUNE}/tools/d2l_control")
import scoring_core  # noqa: E402
from episodes import build_rune_episodes, extract_review_feedback  # noqa: E402

from ctx_to_lora.data.processing import tokenize_ctx_text  # noqa: E402
from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

CKPT = os.environ.get("D2L_CKPT", "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin")
STEPS = int(os.environ.get("D2L_STEPS", "150"))
LR = float(os.environ.get("D2L_LR", "2e-5"))
LAMBDA = float(os.environ.get("D2L_LAMBDA", "1.0"))
MARGIN = float(os.environ.get("D2L_MARGIN", "1.0"))
TRAIN_CORPUS = "/tmp/rune-corpus/external_codereview.train.jsonl"
VAL_CORPUS = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"
MAX_ANS = 48
NIAH_DOC = ("The grass is green. The sky is blue. Here we go. The special magic number is 4417. "
            "There and back again. The minutes were filed.")
CODE_DOC = ("```python\nRETRY_BUDGET = 7\n\ndef quarkle_resync(payload, attempts):\n"
            "    checksum = (payload * 31 + 17) % 9973\n    return checksum\n```\n")


def ctx_ids_for(model, ctx_tok, doc):
    return torch.tensor(tokenize_ctx_text(dict(context=[doc]), ctx_tok)["ctx_ids"], device=model.device)


def _apply_forward(model, full_ids, ctx_ids):
    model.reset()
    model.patch_lora_forward()
    return model(ctx_ids=ctx_ids, ctx_attn_mask=torch.ones_like(ctx_ids),
                 n_ctx_chunks=torch.tensor([1], device=model.device),
                 n_queries=torch.tensor([1], device=model.device), input_ids=full_ids).logits[0]


def _zero_forward(model, full_ids):
    model.reset()
    return model.base_model(input_ids=full_ids).logits[0]


def _full(tok, model, query, answer):
    p = tok.apply_chat_template([{"role": "user", "content": query}], add_special_tokens=False,
                                add_generation_prompt=True, return_tensors="pt").to(model.device)
    a = tok(answer, add_special_tokens=False).input_ids[:MAX_ANS]
    a = torch.tensor([a], device=model.device)
    return torch.cat([p, a], dim=1), p.shape[1], a.shape[1]


def lp_tensor(logits, full, s, n):
    """DIFFERENTIABLE mean gold logprob over the answer span (next-token convention)."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    rows = torch.arange(s - 1, s - 1 + n, device=logits.device)
    gold = full[0, s:s + n]
    return lp[rows, gold].mean()


def swap_feedback(doc, this_fb, other_fb):
    if this_fb and other_fb and this_fb in doc:
        return doc.replace(this_fb, other_fb)
    return None


def eval_diff(model, tok, ctx_tok, val_eps):
    """diff m-zero, m-mismatch(generic other ep), m-mismatch(feedback-swap hardneg)."""
    mz, mm_gen, mm_hard = [], [], []
    for i, e in enumerate(val_eps):
        q = e.queries["diff"]
        full, s, n = _full(tok, model, q["query"], q["answer"])
        if n < 1:
            continue
        with torch.no_grad():
            m = scoring_core.mean_gold_logprob(_apply_forward(model, full, ctx_ids_for(model, ctx_tok, e.doc)), full[0], s, n)
            z = scoring_core.mean_gold_logprob(_zero_forward(model, full), full[0], s, n)
            other = val_eps[(i + 1) % len(val_eps)]
            xg = scoring_core.mean_gold_logprob(_apply_forward(model, full, ctx_ids_for(model, ctx_tok, other.doc)), full[0], s, n)
            hn = swap_feedback(e.doc, extract_review_feedback(e.doc), extract_review_feedback(other.doc))
            mm_hard_v = None
            if hn:
                xh = scoring_core.mean_gold_logprob(_apply_forward(model, full, ctx_ids_for(model, ctx_tok, hn)), full[0], s, n)
                mm_hard_v = m - xh
        mz.append(m - z); mm_gen.append(m - xg)
        if mm_hard_v is not None:
            mm_hard.append(mm_hard_v)
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {"diff_mz": avg(mz), "diff_mm_generic": avg(mm_gen), "diff_mm_feedbackswap": avg(mm_hard), "n_hard": len(mm_hard)}


def retention(model, tok, ctx_tok):
    out = {}
    for name, doc, q, a in [("niah", NIAH_DOC, "What is the special magic number?", "4417"),
                            ("code", CODE_DOC, "In the checksum formula, what number is used as the modulus?", "9973")]:
        full, s, n = _full(tok, model, q, a)
        with torch.no_grad():
            m = scoring_core.mean_gold_logprob(_apply_forward(model, full, ctx_ids_for(model, ctx_tok, doc)), full[0], s, n)
            z = scoring_core.mean_gold_logprob(_zero_forward(model, full), full[0], s, n)
        out[f"{name}_mz"] = m - z
    return out


def main():
    sd = torch.load(CKPT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(sd, train=False, use_sequence_packing=False, use_flash_attn=False)
    model.reset()
    tok = get_tokenizer(model.base_model.name_or_path)
    ctx_tok = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.hypernet.parameters():
        p.requires_grad_(True)
    init = [p.detach().float().cpu().clone() for p in model.hypernet.parameters()]
    print(f"steps={STEPS} lr={LR} lambda={LAMBDA} margin={MARGIN}", flush=True)

    train_eps = build_rune_episodes(TRAIN_CORPUS, n=40)
    val_eps = build_rune_episodes(VAL_CORPUS, n=12)
    print(f"train eps={len(train_eps)} val eps={len(val_eps)}", flush=True)

    before = {**eval_diff(model, tok, ctx_tok, val_eps), **retention(model, tok, ctx_tok)}
    print("BEFORE:", {k: round(v, 3) for k, v in before.items()}, flush=True)

    opt = torch.optim.AdamW(model.hypernet.parameters(), lr=LR)
    import random
    rng = random.Random(0)
    nfb = 0
    for step in range(STEPS):
        e = train_eps[rng.randrange(len(train_eps))]
        other = train_eps[rng.randrange(len(train_eps))]
        q = e.queries["diff"]
        full, s, n = _full(tok, model, q["query"], q["answer"])
        if n < 1:
            continue
        hn = swap_feedback(e.doc, extract_review_feedback(e.doc), extract_review_feedback(other.doc))
        logits_m = _apply_forward(model, full, ctx_ids_for(model, ctx_tok, e.doc))
        ce = torch.nn.functional.cross_entropy(logits_m[s - 1:s - 1 + n].float(), full[0, s:s + n])
        loss = ce
        if hn:
            lp_m = lp_tensor(logits_m, full, s, n)
            lp_neg = lp_tensor(_apply_forward(model, full, ctx_ids_for(model, ctx_tok, hn)), full, s, n)
            hinge = torch.relu(MARGIN - (lp_m - lp_neg))
            loss = ce + LAMBDA * hinge
            nfb += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 25 == 0 or step == STEPS - 1:
            print(f"  step {step:3d} ce {ce.item():.3f} loss {loss.item():.3f} (fb-neg used {nfb})", flush=True)

    after = {**eval_diff(model, tok, ctx_tok, val_eps), **retention(model, tok, ctx_tok)}
    # ||Δhypernet|| / ||hypernet_init||
    num = sum(((p.detach().float().cpu() - i) ** 2).sum().item() for p, i in zip(model.hypernet.parameters(), init)) ** 0.5
    den = sum((i ** 2).sum().item() for i in init) ** 0.5
    rel = num / den
    print("AFTER :", {k: round(v, 3) for k, v in after.items()}, flush=True)
    print("\n=== CONTRASTIVE DIFF ABLATION (before -> after) ===", flush=True)
    for k in before:
        print(f"  {k:22s} {before[k]:+.3f} -> {after[k]:+.3f}  (Δ {after[k]-before[k]:+.3f})", flush=True)
    print(f"  ||Δhypernet||/||init|| = {rel:.4f}  (how 'light' the finetune was)", flush=True)
    import json
    json.dump({"before": before, "after": after, "rel_weight_delta": rel,
               "steps": STEPS, "lr": LR, "lambda": LAMBDA, "margin": MARGIN},
              open("/tmp/d2l_contrastive_result.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
