"""Fast, no-training diagnostic: does the hypernet map DIFFERENT contexts to
DIFFERENT adapters, and is there any needle logprob signal?

Answers the question raised by Stage-0 run #1 (real==contra byte-identical
outputs): is the recall failure about scaling/training/decoding, or does the
context encoder simply not condition on the trajectory (adapters cosine ~ 1.0)?

  - weight sensitivity: pairwise cosine / relative-L2 of generated A,B across
    distinct contexts. cosine ~ 1.0 => hypernet ignores context (deep problem).
  - needle logprob: teacher-forced avg logprob of the needle tokens given the
    answer prefix, under real vs zero vs contradictory context, swept over
    scaling. real >> zero/contra => content signal exists (suppressed by
    decoding in the free-form probe).

Run under tools/run_guarded.sh. GPU-only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rune.config import load_rune_config
from rune.model.hypernetwork import (
    HypernetworkConfig,
    load_hypernetwork,
    reinit_scaler_b_nonzero,
)
from rune.training.hypernet_distill import _functional_lora, _generate_lora_dict

CTXS = {
    "A_73921": "Internal note: MAGIC_OFFSET = 73921 for the payload.",
    "B_11111": "Internal note: MAGIC_OFFSET = 11111 for the payload.",
    "C_zorblax": "Config: ZORBLAX_LIMIT is set to 48207 in production.",
}
PREFIX = "The MAGIC_OFFSET value is"
NEEDLE = " 73921"


def _flatten(lora_dict: dict[str, Any]) -> Any:
    parts = []
    for w in lora_dict.values():
        parts.append(w["A"].detach().float().reshape(-1))
        parts.append(w["B"].detach().float().reshape(-1))
    return torch.cat(parts)


def _needle_logprob(
    base, tok, hypernet, context, prefix, needle, layer_indices, scaling, max_len
) -> float:
    device = next(base.parameters()).device
    with torch.no_grad():
        ld = _generate_lora_dict(hypernet, context, base, tok, layer_indices, max_len)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        n_ids = tok(needle, add_special_tokens=False)["input_ids"]
        full = torch.tensor([p_ids + n_ids], device=device)
        n_qs = torch.tensor([1], device=device)
        ctx = (
            _functional_lora(base, layer_indices, ld, scaling, n_qs)
            if scaling > 0
            else _null()
        )
        with ctx:
            logits = base(full, use_cache=False).logits[0]  # [T, V]
        lp = torch.log_softmax(logits.float(), dim=-1)
        total = 0.0
        for i, tid in enumerate(n_ids):
            pos = len(p_ids) + i - 1  # logit predicting token at len(p_ids)+i
            total += float(lp[pos, tid])
        return total / max(len(n_ids), 1)


def _null():
    return contextlib.nullcontext()


def _degeneration(
    base, tok, hypernet, context, prompt, layer_indices, scaling, max_len
) -> float:
    """1 - distinct/total over a short greedy continuation. High => degenerate.

    Measures preservation/non-degeneration at a given scaling (reviewer point 3):
    a content-independent broad perturbation collapses into repetition.
    """
    device = next(base.parameters()).device
    with torch.no_grad():
        ld = _generate_lora_dict(hypernet, context, base, tok, layer_indices, max_len)
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        n_qs = torch.tensor([1], device=device)
        ctx = (
            _functional_lora(base, layer_indices, ld, scaling, n_qs)
            if scaling > 0
            else _null()
        )
        with ctx:
            gen = base.generate(**ids, max_new_tokens=24, do_sample=False)
        new = gen[0, ids["input_ids"].shape[1] :].tolist()
        if not new:
            return 0.0
        return 1.0 - len(set(new)) / len(new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reinit", action="store_true", help="reinit scaler_B=1 (match training)"
    )
    ap.add_argument(
        "--scalings", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0]
    )
    ap.add_argument(
        "--json-out", type=str, default="/tmp/rune-issue49-sensitivity.json"
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="s3://elixirtrials-949678234935-eu-west-2-artifacts/checkpoints/hypernet_hpo/checkpoint.pt",
    )
    ap.add_argument("--model-id", type=str, default=load_rune_config().model_id)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    base.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    hypernet = load_hypernetwork(
        HypernetworkConfig(checkpoint_path=args.checkpoint), device="cuda"
    )
    if args.reinit:
        reinit_scaler_b_nonzero(hypernet, 1.0)
    hypernet.eval()
    layer_indices = list(hypernet.config.layer_indices)
    max_len = 256

    out: dict[str, Any] = {"reinit": args.reinit, "layer_indices": layer_indices}

    # 1. weight sensitivity across contexts (no training)
    with torch.no_grad():
        vecs = {
            k: _flatten(
                _generate_lora_dict(hypernet, c, base, tok, layer_indices, max_len)
            )
            for k, c in CTXS.items()
        }
    keys = list(vecs)
    cos = {}
    rell2 = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = vecs[keys[i]], vecs[keys[j]]
            cos[f"{keys[i]}|{keys[j]}"] = float(
                torch.nn.functional.cosine_similarity(a, b, dim=0)
            )
            rell2[f"{keys[i]}|{keys[j]}"] = float((a - b).norm() / (b.norm() + 1e-8))
    out["weight_cosine"] = cos
    out["weight_rel_l2"] = rell2

    # 2. needle logprob sweep: real(A) vs contra(B) vs zero
    sweep = []
    for s in args.scalings:
        real = _needle_logprob(
            base,
            tok,
            hypernet,
            CTXS["A_73921"],
            PREFIX,
            NEEDLE,
            layer_indices,
            s,
            max_len,
        )
        contra = _needle_logprob(
            base,
            tok,
            hypernet,
            CTXS["B_11111"],
            PREFIX,
            NEEDLE,
            layer_indices,
            s,
            max_len,
        )
        degen = _degeneration(
            base, tok, hypernet, CTXS["A_73921"], PREFIX, layer_indices, s, max_len
        )
        sweep.append(
            {
                "scaling": s,
                "real_logprob": real,
                "contra_logprob": contra,
                "real_minus_contra": real - contra,
                "degeneration": degen,
            }
        )
    out["needle_logprob_sweep"] = sweep

    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print("WEIGHT_COSINE:", json.dumps(cos))
    print("NEEDLE_SWEEP:", json.dumps(sweep))
    # Interpretation hint (not a gate): context-insensitive if all cosines > 0.98.
    print("CONTEXT_SENSITIVE:", any(v < 0.98 for v in cos.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
