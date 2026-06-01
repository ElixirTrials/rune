"""Issue #52 — retention-gated LIGHT FINETUNE ablation (tests the user's conjecture).

Warm-start Sakana's working Doc2LoRA checkpoint, lightly fine-tune the hypernet on a small
queryable-EPISODE set built from Rune's code-review patches (objective = recall the episode:
goal/file/diff answer-CE, mirroring their CrossEntropyTrainer), then re-score:
  GAIN gate    : Rune-episode facets (goal/file/diff) + continuation improve vs zero-shot.
  RETENTION gate: NIAH magic-number recall + clean code-fact recall are PRESERVED.
A useful specialization improves Rune facets WITHOUT forgetting broad recall (reviewer).
Train rows from train.jsonl; eval episodes from val.clean.jsonl (disjoint). Run from this
dir's venv under run_guarded.sh. D2L_CKPT selects checkpoint; D2L_STEPS / D2L_LR tune.
"""
import os
import sys

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
sys.path.insert(0, f"{RUNE}/tools/d2l_control")
import scoring_core  # noqa: E402
from episodes import build_rune_episodes  # noqa: E402

from ctx_to_lora.data.processing import tokenize_ctx_text  # noqa: E402
from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

CKPT = os.environ.get("D2L_CKPT", "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin")
STEPS = int(os.environ.get("D2L_STEPS", "150"))
LR = float(os.environ.get("D2L_LR", "2e-5"))
TRAIN_CORPUS = "/tmp/rune-corpus/external_codereview.train.jsonl"
VAL_CORPUS = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"
MAX_ANS = 48

# Retention probes (must be preserved): NIAH needle + clean code fact.
NIAH_DOC = ("The grass is green. The sky is blue. Here we go. "
            "The special magic number is 4417. There and back again. The minutes were filed.")
NIAH_Q, NIAH_A = "What is the special magic number?", "4417"
CODE_DOC = ("```python\nRETRY_BUDGET = 7\n\ndef quarkle_resync(payload, attempts):\n"
            "    checksum = (payload * 31 + 17) % 9973\n    return checksum\n```\n")
CODE_Q, CODE_A = "In the checksum formula, what number is used as the modulus?", "9973"


def ctx_ids_for(model, ctx_tok, doc):
    return torch.tensor(tokenize_ctx_text(dict(context=[doc]), ctx_tok)["ctx_ids"], device=model.device)


def _forward_logits(model, full_ids, ctx_ids, train=False):
    model.reset()
    model.patch_lora_forward()
    kw = dict(ctx_ids=ctx_ids, ctx_attn_mask=torch.ones_like(ctx_ids),
              n_ctx_chunks=torch.tensor([1], device=model.device),
              n_queries=torch.tensor([1], device=model.device), input_ids=full_ids)
    if train:
        return model(**kw).logits[0]
    with torch.no_grad():
        return model(**kw).logits[0]


def _zero_logits(model, full_ids):
    model.reset()
    with torch.no_grad():
        return model.base_model(input_ids=full_ids).logits[0]


def _full(tok, model, query, answer):
    p = tok.apply_chat_template([{"role": "user", "content": query}], add_special_tokens=False,
                                add_generation_prompt=True, return_tensors="pt").to(model.device)
    a = tok(answer, add_special_tokens=False).input_ids[:MAX_ANS]
    a = torch.tensor([a], device=model.device)
    return torch.cat([p, a], dim=1), p.shape[1], a.shape[1]


def score(model, tok, ctx_tok, doc, query, answer, mismatch_doc=None):
    """Return (m_zero, m_mismatch_or_None): lift over no-adapter, and episode-specificity."""
    full, s, n = _full(tok, model, query, answer)
    if n < 1:
        return None
    cx = ctx_ids_for(model, ctx_tok, doc)
    m = scoring_core.mean_gold_logprob(_forward_logits(model, full, cx), full[0], s, n)
    z = scoring_core.mean_gold_logprob(_zero_logits(model, full), full[0], s, n)
    mm = None
    if mismatch_doc is not None:
        cxx = ctx_ids_for(model, ctx_tok, mismatch_doc)
        x = scoring_core.mean_gold_logprob(_forward_logits(model, full, cxx), full[0], s, n)
        mm = m - x
    return m - z, mm


def evaluate(model, tok, ctx_tok, val_eps):
    out = {}
    for tgt in ("goal", "file", "diff"):
        mz, mm = [], []
        for i, e in enumerate(val_eps):
            mis = val_eps[(i + 1) % len(val_eps)].doc  # a DIFFERENT episode = mismatch control
            r = score(model, tok, ctx_tok, e.doc, e.queries[tgt]["query"], e.queries[tgt]["answer"], mismatch_doc=mis)
            if r:
                mz.append(r[0]); mm.append(r[1])
        out[f"rune_{tgt}_mz"] = sum(mz) / len(mz)
        out[f"rune_{tgt}_mm"] = sum(mm) / len(mm)  # episode-specificity (the #49-relevant metric)
    out["niah_mz"] = score(model, tok, ctx_tok, NIAH_DOC, NIAH_Q, NIAH_A)[0]
    out["code_mz"] = score(model, tok, ctx_tok, CODE_DOC, CODE_Q, CODE_A)[0]
    return out


def main():
    sd = torch.load(CKPT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(sd, train=False, use_sequence_packing=False, use_flash_attn=False)
    model.reset()
    tok = get_tokenizer(model.base_model.name_or_path)
    ctx_tok = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    # freeze everything except the hypernet
    for p in model.parameters():
        p.requires_grad_(False)
    n_tr = 0
    for p in model.hypernet.parameters():
        p.requires_grad_(True); n_tr += p.numel()
    print(f"trainable hypernet params: {n_tr/1e6:.1f}M | steps={STEPS} lr={LR}", flush=True)

    # build train (query,answer) examples from train.jsonl episodes (goal/file/diff)
    train_eps = build_rune_episodes(TRAIN_CORPUS, n=40)
    train_ex = [(e.doc, q["query"], q["answer"]) for e in train_eps for q in e.queries.values()]
    val_eps = build_rune_episodes(VAL_CORPUS, n=12)
    print(f"train examples={len(train_ex)} from {len(train_eps)} eps | val eps={len(val_eps)}", flush=True)

    before = evaluate(model, tok, ctx_tok, val_eps)
    print("BEFORE:", {k: round(v, 3) for k, v in before.items()}, flush=True)

    opt = torch.optim.AdamW(model.hypernet.parameters(), lr=LR)
    import random
    rng = random.Random(0)
    for step in range(STEPS):
        doc, q, ans = train_ex[rng.randrange(len(train_ex))]
        full, s, n = _full(tok, model, q, ans)
        if n < 1:
            continue
        cx = ctx_ids_for(model, ctx_tok, doc)
        logits = _forward_logits(model, full, cx, train=True)  # [L,V]
        # CE on answer span (next-token: predict token t from row t-1)
        tgt = full[0, s:s + n]
        pred = logits[s - 1:s - 1 + n]
        loss = torch.nn.functional.cross_entropy(pred.float(), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 25 == 0 or step == STEPS - 1:
            print(f"  step {step:3d} loss {loss.item():.3f}", flush=True)

    after = evaluate(model, tok, ctx_tok, val_eps)
    print("\n=== LIGHT-FINETUNE ABLATION: before -> after ===", flush=True)
    for k in before:
        d = after[k] - before[k]
        tag = "RETENTION" if k in ("niah_mz", "code_mz") else ("SPECIFICITY" if k.endswith("_mm") else "lift")
        print(f"  [{tag:11s}] {k:12s} {before[k]:+.3f} -> {after[k]:+.3f}  (Δ {d:+.3f})", flush=True)
    # GAIN = episode-SPECIFICITY (m-mismatch) up, not just m-zero (guards the #49 generic-booster trap)
    gained = all(after[f"rune_{t}_mm"] >= before[f"rune_{t}_mm"] - 0.02 for t in ("goal", "file", "diff"))
    improved_any = any(after[f"rune_{t}_mm"] > before[f"rune_{t}_mm"] + 0.05 for t in ("goal", "file", "diff"))
    retained = (after["niah_mz"] > 0.7 * before["niah_mz"]) and (after["code_mz"] > 0.7 * before["code_mz"])
    print(f"\nSPECIFICITY gate (m-mismatch not worse): {gained} | improved_any: {improved_any} "
          f"| RETENTION gate (NIAH&code >=70% kept): {retained}", flush=True)
    import json
    json.dump({"before": before, "after": after, "steps": STEPS, "lr": LR,
               "gained": gained, "improved_any": improved_any, "retained": retained},
              open("/tmp/d2l_ft_result.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
