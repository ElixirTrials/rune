"""Path A/B (Rune side): qwen_4b_d2l recall via RUNE's functional path.

Twin of third_party/doc-to-lora/rune_episode_recall.py (the Sakana-internalize side).
IDENTICAL episodes (build_rune_episodes), IDENTICAL QA query/answer construction
(chat template + generation prompt + answer[:48]), IDENTICAL scoring
(scoring_core.mean_gold_logprob), IDENTICAL matched/mismatch/zero protocol.

The ONLY difference vs the Sakana side is HOW the adapter is produced + applied:
  Rune path = extract_activations_with_model(doc)
            [full base, all layer_indices, output_hidden_states]
            -> hypernet.generate_weights(features, attn_mask, None)
            -> _functional_lora(base, li, lora_dict, scaling)
            [raw B@A*scaling, NO head bias]
  (Sakana path = ctx_encoder PerLayerActivations -> perceiver
   -> combine_lora WITH bias -> alpha/r)

So Sakana(+~2.3) vs this twin isolates whether RUNE'S PATH is guilty (feature
extraction + no-bias functional application), holding episodes/queries/scoring
constant.
Run in RUNE's venv: uv run python tools/_pathab_rune.py [--scaling 5.66]
"""

from __future__ import annotations

import argparse
import sys

import torch

from rune.config import load_rune_config

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
sys.path.insert(0, f"{RUNE}/tools/d2l_control")

import scoring_core  # noqa: E402
from episodes import build_rune_episodes  # noqa: E402

CKPT = (
    f"{RUNE}/third_party/doc-to-lora/trained_d2l/"
    "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
BASE = load_rune_config().model_id
CORPUS = "/tmp/rune-corpus/external_codereview.val.clean.jsonl"
MAX_ANS_TOK = 48
N_EPISODES = 12


def build_full(tok, device, query: str, answer: str):
    """Same chat-format prompt+answer as the Sakana side's build_full."""
    chat = [{"role": "user", "content": query}]
    enc = tok.apply_chat_template(
        chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    )
    p = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
    a = tok(answer, add_special_tokens=False).input_ids[:MAX_ANS_TOK]
    a = torch.tensor([a], device=device)
    full = torch.cat([p, a], dim=1)
    return full, p.shape[1], a.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    # default None -> shared contract effective_scaling(hyp)==lora_alpha (the fix);
    # pass a comma-list (e.g. "5.66,45.25") to sweep explicit values instead.
    ap.add_argument("--scaling", type=str, default=None)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="load base in bf16 (no 4-bit) — match Sakana",
    )
    ap.add_argument("--arms", type=str, default="raw,combined")
    a = ap.parse_args()
    ckpt, base_id = a.ckpt, a.model_id

    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from rune.model.adapter_contract import effective_scaling  # noqa: PLC0415
    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        extract_activations_with_model,
        load_hypernetwork,
    )
    from rune.training.hypernet_distill import _functional_lora  # noqa: PLC0415

    print("building rune episodes...", flush=True)
    eps = build_rune_episodes(CORPUS, n=N_EPISODES)
    print(f"  {len(eps)} episodes", flush=True)
    if len(eps) < 2:
        print("not enough episodes")
        return 1

    print("loading base (4bit) + hypernet (Rune path)...", flush=True)
    load_kw = dict(
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    )
    if not a.bf16:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    print(f"  base dtype = {'bf16' if a.bf16 else '4bit-nf4'}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, **load_kw).eval()
    tok = AutoTokenizer.from_pretrained(base_id)
    hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=ckpt), device="cuda")
    hyp.eval()
    # Resolve to the shared contract (lora_alpha) unless an explicit sweep was given.
    if a.scaling is None:
        a.scaling = str(effective_scaling(hyp))
    li = [int(x) for x in hyp.config.layer_indices]
    hyp_device = next(hyp.parameters()).device
    hyp_dtype = next(hyp.parameters()).dtype
    device = next(base.parameters()).device

    def lora_dict_for(doc: str):
        feats, am = extract_activations_with_model(doc, base, tok, li, a.max_seq_length)
        feats = feats.to(device=hyp_device, dtype=hyp_dtype)
        am = am.to(hyp_device)
        with torch.no_grad():
            ld, _ = hyp.generate_weights(feats, am, None)
        return ld

    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415

    n_chunks = torch.tensor([1], device=device)
    n_qs = torch.tensor([1], device=device)
    docs = [e.doc for e in eps]
    raw = [lora_dict_for(d) for d in docs]  # context-dependent rank-8 A/B
    head_bias = hyp.get_head_bias() if getattr(hyp.config, "use_bias", False) else None

    # DRY: the two arms differ ONLY in adapter ASSEMBLY; everything else (episodes,
    # queries, scoring, functional application) is shared. Arm "combined" reuses
    # ctx_to_lora.combine_lora (the same assembly the engine's
    # generate_adapter_weights uses) + the head bias, then the SAME
    # _functional_lora applies it.
    def assemble(arm: str, d):
        if arm == "raw":
            return d
        return combine_lora(d, n_chunks, lora_bias=head_bias)

    def logits_with(ld, full):
        with torch.no_grad(), _functional_lora(base, li, ld, a.scaling, n_qs):
            return base(full, use_cache=False).logits[0]

    scalings = [float(s) for s in str(a.scaling).split(",")]
    arms = tuple(a.arms.split(","))
    for sc in scalings:
        a.scaling = sc
        for arm in arms:
            asm = [assemble(arm, d) for d in raw]
            per_target: dict[str, list[tuple[float, float]]] = {
                "goal": [],
                "file": [],
                "diff": [],
            }
            for i, e in enumerate(eps):
                mis = asm[(i + 1) % len(eps)]
                for tname, qd in e.queries.items():
                    full, start, length = build_full(
                        tok, device, qd["query"], qd["answer"]
                    )
                    if length < 1:
                        continue
                    ids = full[0]
                    lp_m = scoring_core.mean_gold_logprob(
                        logits_with(asm[i], full), ids, start, length
                    )
                    lp_x = scoring_core.mean_gold_logprob(
                        logits_with(mis, full), ids, start, length
                    )
                    with torch.no_grad():
                        lp_z = scoring_core.mean_gold_logprob(
                            base(full, use_cache=False).logits[0], ids, start, length
                        )
                    per_target[tname].append((lp_m - lp_x, lp_m - lp_z))

            bias_state = (
                "on" if (arm == "combined" and head_bias is not None) else "off"
            )
            print(
                f"\n=== RUNE-FUNCTIONAL ARM={arm} (scaling={a.scaling}, "
                f"bias={bias_state}) ===",
                flush=True,
            )
            allmm = []
            for t, vals in per_target.items():
                if not vals:
                    continue
                mm = sum(v[0] for v in vals) / len(vals)
                mz = sum(v[1] for v in vals) / len(vals)
                spec = sum(1 for v in vals if v[0] > 0) / len(vals)
                allmm += [v[0] for v in vals]
                print(
                    f"  {t:5s} n={len(vals):2d} m-mismatch={mm:+.3f} m-zero={mz:+.3f}"
                    f"frac(m-mis>0)={spec:.2f}",
                    flush=True,
                )
            if allmm:
                print(
                    f"  OVERALL m-mismatch={sum(allmm) / len(allmm):+.3f}  "
                    "(Sakana-internalize ref +1.604 / goal +2.235 on same episodes)",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
