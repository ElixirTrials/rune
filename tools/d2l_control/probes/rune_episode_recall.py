"""Issue #52 — ISOLATION TEST: run RUNE's own code-review episodes through Sakana's
working Doc2LoRA checkpoint.

If Sakana's checkpoint recalls Rune-episode facts (goal / file / diff) with
m-mismatch > 0, then Rune's failure (#49: m-mismatch ~0) is NOT the architecture, the
probe, or the base model's ability to bind these facts -- it is Rune's TRAINING
(objective / data / scale). This is the decisive "which difference matters" experiment.

Episodes come from rune's tools/d2l_control/episodes.build_rune_episodes (reformulated
external_codereview rows: doc = activation_text [code + feedback]; queries = goal
[feedback], file [path], diff [pre->post hunk], each answer an exact span). Run from this
dir's venv. Pure-torch scoring_core is imported from rune's tools/ via sys.path.
"""
import sys

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")            # scoring_core
sys.path.insert(0, f"{RUNE}/tools/d2l_control")  # episodes

import scoring_core  # noqa: E402
from episodes import build_rune_episodes  # noqa: E402

from ctx_to_lora.data.processing import tokenize_ctx_text  # noqa: E402
from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

import os
CKPT = os.environ.get("D2L_CKPT", "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin")
CORPUS = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"
MAX_ANS_TOK = 48  # cap long diff/goal spans so the metric stays comparable
N_EPISODES = 12


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


def build_full(tok, model, query, answer):
    chat = [{"role": "user", "content": query}]
    p = tok.apply_chat_template(
        chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    a = tok(answer, add_special_tokens=False).input_ids[:MAX_ANS_TOK]
    a = torch.tensor([a], device=model.device)
    full = torch.cat([p, a], dim=1)
    return full, p.shape[1], a.shape[1]


def main():
    print("building rune episodes...", flush=True)
    eps = build_rune_episodes(CORPUS, n=N_EPISODES)
    print(f"  {len(eps)} episodes (each carries goal/file/diff)", flush=True)
    if len(eps) < 2:
        print("not enough episodes", flush=True)
        return 1

    print("loading sakana checkpoint...", flush=True)
    sd = torch.load(CKPT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        sd, train=False, use_sequence_packing=False, use_flash_attn=False
    )
    model.reset()
    tok = get_tokenizer(model.base_model.name_or_path)
    ctx_tok = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    ctxs = [ctx_ids_for(model, ctx_tok, e.doc) for e in eps]
    per_target = {"goal": [], "file": [], "diff": []}
    for i, e in enumerate(eps):
        mis = ctxs[(i + 1) % len(eps)]
        for tname, q in e.queries.items():
            full, start, length = build_full(tok, model, q["query"], q["answer"])
            if length < 1:
                continue
            lp_m = scoring_core.mean_gold_logprob(logits_with_ctx(model, full, ctxs[i]), full[0], start, length)
            lp_x = scoring_core.mean_gold_logprob(logits_with_ctx(model, full, mis), full[0], start, length)
            lp_z = scoring_core.mean_gold_logprob(logits_zero(model, full), full[0], start, length)
            per_target[tname].append((lp_m - lp_x, lp_m - lp_z))
            print(f"[ep{i}:{tname}] len={length} m={lp_m:.3f} mis={lp_x:.3f} zero={lp_z:.3f} "
                  f"| m-mis={lp_m-lp_x:+.3f} m-zero={lp_m-lp_z:+.3f}", flush=True)

    print("\n=== RUNE-EPISODES-THROUGH-SAKANA SUMMARY ===", flush=True)
    allmm = []
    for t, vals in per_target.items():
        if not vals:
            continue
        mm = sum(v[0] for v in vals) / len(vals)
        mz = sum(v[1] for v in vals) / len(vals)
        spec = sum(1 for v in vals if v[0] > 0) / len(vals)
        allmm += [v[0] for v in vals]
        print(f"  {t:5s} n={len(vals):2d}  mean m-mismatch={mm:+.3f}  mean m-zero={mz:+.3f}  frac(m-mis>0)={spec:.2f}", flush=True)
    overall = sum(allmm) / len(allmm)
    print(f"  OVERALL mean m-mismatch={overall:+.3f}  (Sakana code-fact ref ~+7.1; Rune #49 ~+0.01)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
