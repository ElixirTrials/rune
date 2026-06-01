"""Stage-0 discriminator for issue #49: can the D2L loop make a non-inert,
content-retrieving adapter on an oracle-free synthetic needle corpus?

Three phases, one model load:
  PHASE 0 (contract): one forward+backward proving the adapter (a) APPLIES
    (student logits differ from base) and (b) is DIFFERENTIABLE (scaler_B receives
    a non-zero gradient). This is the go/no-go before spending steps on training.
  PHASE 1 (train): overfit the hypernet on 3-5 unguessable-needle records.
  PHASE 2 (recall): held-out recall — real vs zero vs contradictory adapter.

Gate: real_hit_rate > zero_hit_rate AND real_hit_rate > contradictory_hit_rate.

Run under tools/run_guarded.sh (15GB CPU box, offload_base=False). GPU-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rune.model.hypernetwork import (
    HypernetworkConfig,
    load_hypernetwork,
    reinit_scaler_b_nonzero,
)
from rune.training.collapse_metrics import (
    assert_optimizer_covers,
    diff_agreement,
    summarize_named_tensors,
)
from rune.training.hypernet_distill import (
    _functional_lora,
    _generate_lora_dict,
    _grad_norm_summary,
    _student_logits,
    _teacher_base_logits,
    distill_step_loss,
)

# Needles live ONLY in the context; held-out prompts omit them.
RECORDS = [
    {
        "needle": "73921",
        "context": "Internal note: MAGIC_OFFSET = 73921 for the payload.",
        "answer": "The MAGIC_OFFSET value is 73921.",
    },
    {
        "needle": "frobnicate",
        "context": "The handler to call on startup is named frobnicate.",
        "answer": "The startup handler is called frobnicate.",
    },
    {
        "needle": "48207",
        "context": "Config: ZORBLAX_LIMIT is set to 48207 in production.",
        "answer": "The ZORBLAX_LIMIT is 48207.",
    },
    {
        "needle": "qux",
        "context": "The secret access token for the vault is qux.",
        "answer": "The vault access token is qux.",
    },
]
# Contradictory contexts: same answer template, WRONG needle value.
CONTRA = [
    "Internal note: MAGIC_OFFSET = 11111 for the payload.",
    "The handler to call on startup is named gizmo.",
    "Config: ZORBLAX_LIMIT is set to 99999 in production.",
    "The secret access token for the vault is blarg.",
]


def _answer_prefix(answer: str, needle: str) -> str:
    """Everything in the answer up to (but excluding) the needle."""
    return answer[: answer.index(needle)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scaling", type=float, default=2.0)
    ap.add_argument("--json-out", type=str, default="/tmp/rune-issue49-synth.json")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="s3://elixirtrials-949678234935-eu-west-2-artifacts/checkpoints/hypernet_hpo/checkpoint.pt",
    )
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-9B")
    args = ap.parse_args()

    out: dict[str, Any] = {"phase": "init"}
    print(
        "free -g:\n"
        + subprocess.run(
            ["free", "-g"], capture_output=True, text=True, check=False
        ).stdout
    )

    device = "cuda"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model_id)

    hypernet = load_hypernetwork(
        HypernetworkConfig(checkpoint_path=args.checkpoint), device=device
    )
    reinit_scaler_b_nonzero(hypernet, 1.0)
    hypernet.train()
    layer_indices = list(hypernet.config.layer_indices)
    out["layer_indices"] = layer_indices
    out["target_modules"] = list(hypernet.config.lora_config.target_modules)

    watched: dict[str, Any] = {}
    if hasattr(hypernet, "scaler_B"):
        first = next(iter(hypernet.scaler_B.keys()))
        watched["scaler_B"] = hypernet.scaler_B[first]

    max_len = 256

    # ---- PHASE 0: adapter-application contract -----------------------------
    r0 = RECORDS[0]
    lora_dict = _generate_lora_dict(
        hypernet, r0["context"], base, tok, layer_indices, max_len
    )
    teacher, base_logits, ans_ids = _teacher_base_logits(
        base, tok, r0["context"], r0["answer"], max_len
    )
    student = _student_logits(
        base, tok, ans_ids, lora_dict, layer_indices, args.scaling
    )
    # base-no-adapter logits over the same answer span (scaling=0 -> adapter off):
    student_off = _student_logits(base, tok, ans_ids, lora_dict, layer_indices, 0.0)
    applies = float((student - student_off).abs().max())
    labels = torch.ones(teacher.shape[0], dtype=torch.long, device=device)
    loss0 = distill_step_loss(
        student,
        teacher,
        base_logits.argmax(-1),
        teacher.argmax(-1),
        labels,
        k=hypernet_topk(),
    )
    grad_ok = False
    sb_grad = 0.0
    if loss0.requires_grad:
        loss0.backward()
        if watched and watched["scaler_B"].grad is not None:
            sb_grad = float(watched["scaler_B"].grad.abs().sum())
            grad_ok = sb_grad > 0.0
    out["contract"] = {
        "adapter_applies": applies > 1e-4,
        "adapter_delta_absmax": applies,
        "loss_requires_grad": bool(loss0.requires_grad),
        "scaler_b_grad_abs_sum": sb_grad,
        "grad_flows": grad_ok,
    }
    print("CONTRACT:", json.dumps(out["contract"]))
    if not (out["contract"]["adapter_applies"] and grad_ok):
        out["phase"] = "contract_failed"
        _dump(args.json_out, out)
        print(
            "CONTRACT FAILED — not a science result, fix plumbing. Aborting before training."
        )
        return 2

    # ---- PHASE 1: overfit --------------------------------------------------
    for p in hypernet.parameters():
        if p.grad is not None:
            p.grad = None
    opt = torch.optim.AdamW(
        [p for p in hypernet.parameters() if p.requires_grad], lr=args.lr
    )
    assert_optimizer_covers(watched, opt)
    metrics: list[dict[str, Any]] = []
    step = 0
    while step < args.max_steps:
        for rec in RECORDS:
            if step >= args.max_steps:
                break
            ld = _generate_lora_dict(
                hypernet, rec["context"], base, tok, layer_indices, max_len
            )
            t, b, ans_ids = _teacher_base_logits(
                base, tok, rec["context"], rec["answer"], max_len
            )
            s = _student_logits(base, tok, ans_ids, ld, layer_indices, args.scaling)
            lab = torch.ones(t.shape[0], dtype=torch.long, device=device)
            loss = distill_step_loss(
                s, t, b.argmax(-1), t.argmax(-1), lab, k=hypernet_topk()
            )
            if not loss.requires_grad:
                step += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in hypernet.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            if step % 10 == 0:
                m = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "diff_agreement": diff_agreement(
                        s.argmax(-1), t.argmax(-1), b.argmax(-1)
                    ),
                    **summarize_named_tensors(watched),
                    **_grad_norm_summary(hypernet),
                }
                metrics.append(m)
                print("step", json.dumps(m))
            step += 1
            del ld, t, b, s
            torch.cuda.empty_cache()
    out["train_metrics"] = metrics

    # ---- PHASE 2: held-out recall -----------------------------------------
    hypernet.eval()
    real_hits = zero_hits = contra_hits = 0
    details = []
    for i, rec in enumerate(RECORDS):
        prefix = _answer_prefix(rec["answer"], rec["needle"])
        real = _recall(
            base,
            tok,
            hypernet,
            rec["context"],
            prefix,
            layer_indices,
            args.scaling,
            max_len,
        )
        zero = _recall(
            base, tok, hypernet, rec["context"], prefix, layer_indices, 0.0, max_len
        )
        contra = _recall(
            base, tok, hypernet, CONTRA[i], prefix, layer_indices, args.scaling, max_len
        )
        rh = rec["needle"] in real
        zh = rec["needle"] in zero
        ch = rec["needle"] in contra
        real_hits += rh
        zero_hits += zh
        contra_hits += ch
        details.append(
            {
                "needle": rec["needle"],
                "real": real,
                "zero": zero,
                "contra": contra,
                "real_hit": rh,
                "zero_hit": zh,
                "contra_hit": ch,
            }
        )
    n = len(RECORDS)
    out["recall"] = {
        "real_hit_rate": real_hits / n,
        "zero_hit_rate": zero_hits / n,
        "contradictory_hit_rate": contra_hits / n,
        "details": details,
    }
    out["gate_passed"] = (real_hits > zero_hits) and (real_hits > contra_hits)
    out["phase"] = "done"
    _dump(args.json_out, out)
    print(
        "GATE:",
        "PASS" if out["gate_passed"] else "FAIL",
        json.dumps(
            {
                k: out["recall"][k]
                for k in ("real_hit_rate", "zero_hit_rate", "contradictory_hit_rate")
            }
        ),
    )
    return 0 if out["gate_passed"] else 1


def hypernet_topk() -> int:
    return 50


def _recall(
    base, tok, hypernet, context, prefix, layer_indices, scaling, max_len
) -> str:
    device = next(base.parameters()).device
    with torch.no_grad():
        ld = _generate_lora_dict(hypernet, context, base, tok, layer_indices, max_len)
        ids = tok(prefix, add_special_tokens=False, return_tensors="pt").to(device)
        n_qs = torch.tensor([1], device=device)
        with _functional_lora(base, layer_indices, ld, scaling, n_qs):
            gen = base.generate(**ids, max_new_tokens=8, do_sample=False)
        return tok.decode(gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


def _dump(path: str, obj: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
