"""E1 oracle per-episode LoRA capacity probe (issue #52, lead discriminator).

UPPER BOUND on the SAME substrate as the hypernet. For each of the 10 frozen MBPP
episodes used by tools/_specificity_probe.py, fit an ORACLE PEFT LoRA by CE on the
episode ANSWER span only, then score the trained oracle on the BODY span [hi, len)
through the IDENTICAL mask + math as the hypernet (reusing _specificity_probe's
span_bounds + scoring_core.mean_gold_logprob).

This answers E1's capacity-vs-representation question (spec §86-109):
  - oracle good @ r8 + hypernet bad @ r8  -> REPRESENTATION wall.
  - oracle bad @ r8, good @ higher rank   -> CAPACITY  (use --rank 16 / 32).
  - both bad @ high rank                  -> DATA / ARCHITECTURE.

FROZEN contract (issue52-predeclared-spec-T0-E1-E2-2026-06-02.md):
  - Oracle = peft.get_peft_model(base, LoraConfig(r=RANK,
    target_modules=['down_proj'], lora_alpha=RANK*45.2548, lora_dropout=0.0)).
    lora_alpha = RANK*45.2548 because PEFT applies alpha/r while the hypernet/
    functional path applies lora_alpha UN-DIVIDED (adapter_contract.py:54); so
    (RANK*45.2548)/RANK == 45.2548 == effective_scaling at every rank -> identical
    substrate scale on the capacity branch too.
  - target_modules MUST be ['down_proj'] (PEFT defaults to attn q/v = over-capacity).
    down_proj x 36 layers (Qwen3-4B) = the hypernet's layer_indices.
  - Train CE on the answer span only, answer-preserving truncation via
    rune.training.hypernet_distill._prepare_ids at max_seq_length=768.
  - Score the trained oracle through the IDENTICAL BODY mask + math as the hypernet:
    BODY = [hi, len) where [lo, hi) is the def-<entry_point>( signature line.
    NEVER signature, NEVER full-span (signature m-mismatch +3.84 dwarfs body +0.14).
  - 4-bit nf4, bf16 compute, double-quant; flash_attention_2; device_map {'':'cuda'}.
  - gradient_checkpointing MAY be True for the oracle (PEFT-native forward).

Arms (mirror _specificity_probe.py exactly): per episode i, train oracle adapter_i.
  matched  = adapter_i scored on episode_i answer.
  mismatch = adapter_{perm(i)} (derangement partner; reuse the already-trained
             partner adapter, no retrain) scored on episode_i answer.
  zero     = base, adapter disabled, scored on episode_i answer.

This file TRAINS on GPU when the main loop runs it. GPU imports are deferred inside
main() so the module is importable / ast-parseable on CPU-only CI.

Run (per rank, via the RAM watchdog):
  tools/run_guarded.sh /tmp/e1_oracle_r8.log  tools/_e1_oracle.py --rank 8  --out /tmp/e1_oracle_r8.jsonl
  tools/run_guarded.sh /tmp/e1_oracle_r16.log tools/_e1_oracle.py --rank 16 --out /tmp/e1_oracle_r16.jsonl
  tools/run_guarded.sh /tmp/e1_oracle_r32.log tools/_e1_oracle.py --rank 32 --out /tmp/e1_oracle_r32.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
sys.path.insert(0, f"{RUNE}/tools/d2l_control")

import scoring_core  # noqa: E402

# Reuse the EXACT frozen apparatus from the hypernet probe so the oracle and the
# hypernet share byte-identical episodes, references, derangement, BODY-span math,
# the ABSENT (hidden-regime) prompt, and the MAX_ANS_TOK cap. Importing this module
# does NO GPU work at import time (torch import only).
from _specificity_probe import (  # noqa: E402
    ABSENT,
    BASE,
    MAX_ANS_TOK,
    REFS,
    TASKS_FILE,
    build_full,
    derangement,
    span_bounds,
)

# Hypernet/functional path applies lora_alpha UN-DIVIDED (adapter_contract.py:54).
# An oracle PEFT LoRA matches the substrate scale when lora_alpha = RANK*EFFECTIVE,
# because PEFT divides by r: (RANK*EFFECTIVE)/RANK == EFFECTIVE at every rank.
EFFECTIVE_SCALING = 45.2548
MAX_SEQ_LENGTH = 768


def body_span(tok, ans: str, entry_point: str, start: int, length: int):
    """BODY span (start_pos, span_len) for scoring, with FROZEN marker hardening.

    Reuses _specificity_probe.span_bounds (the IDENTICAL mask). The signature
    occupies answer-token range [lo, hi); BODY = [hi, length). Spec §92: the (0,0)
    missing-marker fallback silently makes BODY = the FULL answer incl. signature
    (the +3.84 signature contaminant), collapsing the discriminator. HARDEN: any
    episode that hits the (0,0) fallback (hi <= lo) or has an empty body
    (length <= hi) is EXCLUDED with an explicit reason, never scored under (0,0).

    Returns ((body_start, body_len), None) on success, else (None, reason).
    """
    lo, hi = span_bounds(tok, ans, entry_point)
    if hi <= lo:
        return None, "marker_not_found"  # (0,0) fallback -> would contaminate BODY
    if length <= hi:
        return None, "empty_body"  # body span [hi, length) is empty
    return (start + hi, length - hi), None


def train_oracle(base, tok, init_sd, set_sd_fn, get_sd_fn, prompt: str,
                 answer: str, steps: int, lr: float, device):
    """Fit the current PEFT adapter on ONE episode by CE on the answer span only.

    TRAIN SURFACE == SCORE SURFACE (advisor). Train on the SAME ABSENT prompt the
    oracle is scored on (via build_full), supervising only the answer-span tokens
    [start, start+length). The ABSENT prompt carries NO task identity, so the LoRA
    must STORE the answer in its rank-r weights to recover it — a TRUE capacity test,
    not a context-transfer artifact (training on context+answer then scoring on
    ABSENT would make a weak oracle uninterpretable: capacity wall vs surface-mismatch
    transfer failure). Overfitting the single short answer is the GOAL (oracle =
    capacity upper bound). Returns the trained PEFT-LoRA state_dict.

    Per-episode independence: restore the PRISTINE init adapter (`init_sd`, captured
    once right after get_peft_model, with B=0 == identity adapter) before each
    episode via set_sd_fn. Deterministic and version-stable, unlike a
    reset_lora_parameters() call whose silent no-op would let adapters accumulate.
    """
    full, start, length = build_full(tok, device, prompt, answer)
    if length < 1:
        raise ValueError("empty answer span")
    labels = torch.full_like(full, -100)
    labels[0, start:start + length] = full[0, start:start + length]  # answer span only

    set_sd_fn(base, init_sd)  # restore pristine (identity) adapter

    trainable = [p for p in base.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    base.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = base(input_ids=full, labels=labels, use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    base.eval()
    # Deep-copy the LoRA-only state dict to CPU so the partner adapter survives
    # later set_peft_model_state_dict swaps.
    sd = get_sd_fn(base)
    return {k: v.detach().to("cpu").clone() for k, v in sd.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8,
                    help="oracle LoRA rank (8 default; 16/32 for the capacity branch)")
    ap.add_argument("--steps", type=int, default=200,
                    help="CE steps per episode (overfit the single short answer)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument("--out", type=str, required=True,
                    help="per-episode JSONL output path")
    a = ap.parse_args()

    if a.max_seq_length != MAX_SEQ_LENGTH:
        print(f"[warn] max_seq_length={a.max_seq_length} != frozen {MAX_SEQ_LENGTH}",
              flush=True)

    # ---- deferred GPU imports (CPU-only CI importability) ----
    from peft import (  # noqa: PLC0415
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tasks = json.loads(Path(TASKS_FILE).read_text())
    n = len(tasks)
    perm = derangement(n)
    for t in tasks:
        if t["task_id"] not in REFS:
            raise KeyError(f"no reference solution for {t['task_id']}")

    lora_alpha = a.rank * EFFECTIVE_SCALING  # PEFT alpha/r == EFFECTIVE_SCALING
    print(
        f"E1 oracle: rank={a.rank} lora_alpha={lora_alpha} "
        f"(effective alpha/r={lora_alpha / a.rank}) target_modules=['down_proj'] "
        f"steps={a.steps} lr={a.lr} max_seq_length={a.max_seq_length}",
        flush=True,
    )

    # 4-bit nf4 base (mirror _specificity_probe.py:202-207 / diag_pre_corpus_gate
    # PEFT setup). bf16 compute, double-quant, flash_attention_2, device_map.
    print("loading 4-bit base...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        a.model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    )
    tok = AutoTokenizer.from_pretrained(a.model_id)
    device = next(base.parameters()).device

    # 4-bit + PEFT + gradient_checkpointing: prepare_model_for_kbit_training enables
    # input-require-grads + casts norms so grads flow through the frozen 4-bit base.
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True
    )
    base = get_peft_model(
        base,
        LoraConfig(
            r=a.rank,
            lora_alpha=lora_alpha,
            target_modules=["down_proj"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        ),
    )
    base.print_trainable_parameters()

    # Capture the pristine init adapter (B=0 == identity) ONCE; restored before each
    # episode so the 10 oracles are mutually independent (advisor: deterministic, vs
    # a version-fragile reset_lora_parameters whose silent no-op would collapse the
    # matched-vs-mismatch discriminator).
    init_sd = {
        k: v.detach().to("cpu").clone()
        for k, v in get_peft_model_state_dict(base).items()
    }

    # ---- train + score BOTH on the ABSENT (hidden) prompt: train==score surface ----
    # The ABSENT prompt carries no task identity, so the oracle must STORE each answer
    # in its rank-r weights to recover it -> a true capacity probe, not a context-
    # transfer artifact (advisor). All episodes train on the same ABSENT prompt.
    refs = [REFS[t["task_id"]] for t in tasks]

    # ---- train one oracle adapter per episode (overfit each single answer) ----
    print(f"\ntraining {n} oracle adapters (rank={a.rank})...", flush=True)
    adapters: list[dict] = []
    for i, t in enumerate(tasks):
        sd = train_oracle(
            base, tok, init_sd, set_peft_model_state_dict,
            get_peft_model_state_dict, ABSENT, refs[i], a.steps, a.lr, device,
        )
        adapters.append(sd)
        print(f"  trained oracle {i} ({t['task_id']})", flush=True)

    # ---- score on the BODY span via the IDENTICAL mask + math as the hypernet ----
    # ABSENT (hidden) prompt only: train-surface != score-surface, mirroring the
    # hypernet (conditioned on raw trajectory, scored on the chat-templated ABSENT
    # prompt). Oracle and hypernet then differ ONLY in how the delta was obtained.
    print("\n=== (B) ORACLE BODY-SPAN LOGPROB  regime=absent ===", flush=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float, float]] = []
    excluded: list[tuple[str, str]] = []
    with out_path.open("w") as fout:
        for i, t in enumerate(tasks):
            prompt = ABSENT  # no {desc}: hidden regime, faithful to the hypernet
            ans = refs[i]
            full, start, length = build_full(tok, device, prompt, ans)
            if length < 1:
                excluded.append((t["task_id"], "empty_answer"))
                fout.write(json.dumps({
                    "row_idx": i, "task_id": t["task_id"], "rank": a.rank,
                    "eligible": False, "skip_reason": "empty_answer",
                }) + "\n")
                continue
            ids = full[0]
            span, reason = body_span(tok, ans, t["entry_point"], start, length)
            if span is None:
                excluded.append((t["task_id"], reason))
                fout.write(json.dumps({
                    "row_idx": i, "task_id": t["task_id"], "rank": a.rank,
                    "eligible": False, "skip_reason": reason,
                }) + "\n")
                print(f"  [EXCLUDED] {t['task_id']}: {reason}", flush=True)
                continue
            s, ln = span

            # matched: oracle_i ; mismatch: oracle_{perm(i)} ; zero: base disabled.
            set_peft_model_state_dict(base, adapters[i])
            with torch.no_grad():
                lg_m = base(full, use_cache=False).logits[0]
            set_peft_model_state_dict(base, adapters[perm[i]])
            with torch.no_grad():
                lg_x = base(full, use_cache=False).logits[0]
            with torch.no_grad(), base.disable_adapter():
                lg_z = base(full, use_cache=False).logits[0]

            lp_m = scoring_core.mean_gold_logprob(lg_m, ids, s, ln)
            lp_x = scoring_core.mean_gold_logprob(lg_x, ids, s, ln)
            lp_z = scoring_core.mean_gold_logprob(lg_z, ids, s, ln)
            mm = lp_m - lp_x  # matched - mismatch(derangement)
            mz = lp_m - lp_z  # matched - zero
            rows.append((t["task_id"], mm, mz, lp_m))
            fout.write(json.dumps({
                "row_idx": i, "task_id": t["task_id"], "rank": a.rank,
                "entry_point": t["entry_point"],
                "neg_idx": perm[i], "neg_task_id": tasks[perm[i]]["task_id"],
                "body_start": s, "body_len": ln, "ans_tok": length,
                "lp_matched": lp_m, "lp_mismatch": lp_x, "lp_zero": lp_z,
                "matched_mismatch": mm, "matched_zero": mz,
                "eligible": True, "skip_reason": None,
            }) + "\n")
            print(
                f"    [body] {t['task_id']:8s} lp_m={lp_m:+.3f}(overfit-PC) "
                f"m-mismatch={mm:+.4f}  m-zero={mz:+.4f}",
                flush=True,
            )

        if rows:
            mm_mean = sum(r[1] for r in rows) / len(rows)
            mz_mean = sum(r[2] for r in rows) / len(rows)
            lpm_mean = sum(r[3] for r in rows) / len(rows)
            frac = sum(1 for r in rows if r[1] > 0) / len(rows)
            # Positive control (advisor): the oracle MUST overfit -> matched body
            # logprob near 0. If lpm_mean is low, training was underpowered (raise
            # --steps/--lr) and a low m-mismatch is NOT a capacity verdict. Heuristic
            # bar: mean lp_m > -0.3 nats (~0.74 avg token prob).
            pc_overfit = lpm_mean > -0.3
            summary = {
                "summary": True, "rank": a.rank, "n_scored": len(rows),
                "n_excluded": len(excluded),
                "body_m_mismatch_mean": mm_mean, "body_m_zero_mean": mz_mean,
                "body_lp_matched_mean": lpm_mean, "oracle_overfit_pc": pc_overfit,
                "frac_m_mismatch_pos": frac,
                "excluded": [{"task_id": tid, "reason": r} for tid, r in excluded],
            }
            fout.write(json.dumps(summary) + "\n")
            print(
                f"\n  [body] MEAN n={len(rows):2d} m-mismatch={mm_mean:+.4f}"
                f"  m-zero={mz_mean:+.4f}  frac(m-mis>0)={frac:.2f}"
                f"  excluded={len(excluded)}",
                flush=True,
            )
            print(
                f"  [overfit-PC] mean matched lp_m={lpm_mean:+.4f} -> "
                + ("OK (oracle memorized)" if pc_overfit
                   else "WEAK: raise --steps/--lr; m-mismatch is NOT a capacity verdict"),
                flush=True,
            )
        else:
            print("\n  [body] NO scorable episodes (all excluded)", flush=True)

    print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
