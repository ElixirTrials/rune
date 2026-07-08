"""C4 I5 — capacity curve: K facts in one adapter vs K pointers in the tail.

Extends the frozen C1 keystone instrument (tools/_repobench_clamp_run.py,
byte-untouched, imported as a sibling module) to bundles of K rows on the same
60 keystone rows (levels 8k,32k x 30, offset 100, W=768, seed 0). Arms per K:

  floor          clamped current-file prompt (K-independent, run once)
  tail_k{K}      the bundle's K episodic pointers in the tail; if the joined
                 conditioning alone exceeds W the arm is infeasible and scored
                 as NOT recovered (pre-registered deviation from C1's guard)
  adapter_a_k{K} mode (a): ONE hypernet forward on the K pointers concatenated
  adapter_b_k{K} mode (b): K per-row forwards, rank-stacked composition
  adapter_k1     K=1 (modes coincide) - the C1 anchor arm

Legs: --leg sanity runs floor + adapter_k1 at NATIVE PEFT rank through the
engine path (model.generate_adapter) and must reproduce the C1 run
token-for-token (compare via --c1-traces). --leg capacity loads the PEFT
adapter at campaign_rank(r, bias_rank, K_max) and runs all Ks; per-fact and
mode-(a) adapters are assembled through the probe's low-level path (which the
merge_head_bias_rank guard does not constrain) and zero-padded before hotswap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _c4_capacity_lib as lib  # noqa: E402
import _repobench_clamp_run as clamp  # noqa: E402

C3_SHA256 = "53e24af243a38dfbfad82f7293635bfc592922dd2058fefbbfa10714b5457a3f"
_KS_DEFAULT = "1,2,4,8"
_MARGIN_DEFAULT = 0.15  # proposed; co-author sign-off pre-registered in the plan


# ---------------------------------------------------------------- stats/gate

def _rate(traces: list[dict[str, Any]], label: str) -> tuple[int, int]:
    k = n = 0
    for rec in traces:
        arm = rec["arms"].get(label)
        if arm is None or arm.get("recovered") is None:
            continue
        n += 1
        k += int(bool(arm["recovered"]))
    return k, n


def capacity_metrics(traces: list[dict[str, Any]]) -> dict[str, float]:
    """Per-arm recovery + Wilson CI + infeasible counts + paired McNemars."""
    labels = sorted({lab for rec in traces for lab in rec["arms"]})
    m: dict[str, float] = {}
    for lab in labels:
        k, n = _rate(traces, lab)
        if not n:
            continue
        lo, hi = clamp._wilson_ci(k, n)
        m[f"recovery_{lab}"] = k / n
        m[f"recovery_{lab}_n"] = float(n)
        m[f"recovery_{lab}_wilson_lo"] = lo
        m[f"recovery_{lab}_wilson_hi"] = hi
        m[f"infeasible_{lab}"] = float(sum(
            1 for rec in traces if rec["arms"].get(lab, {}).get("infeasible")
        ))
    for lab in labels:
        if lab == "floor" or lab.startswith("tail_"):
            continue
        for other in ("floor", f"tail_{lab.rsplit('_', 1)[-1]}"):
            if other not in labels:
                continue
            a_only, b_only, n = clamp._paired_discordants(traces, lab, other)
            m[f"mcnemar_{lab}_vs_{other}_first_only"] = float(a_only)
            m[f"mcnemar_{lab}_vs_{other}_second_only"] = float(b_only)
            m[f"mcnemar_{lab}_vs_{other}_n"] = float(n)
            m[f"mcnemar_{lab}_vs_{other}_p"] = clamp._two_sided_binom_p(a_only, b_only)
    return m


def bundle_sign_counts(
    traces: list[dict[str, Any]], arm: str, other: str, k: int
) -> tuple[int, int, int]:
    """Bundle-level sign counts (sensitivity): mean recovery per k-bundle."""
    pos = neg = 0
    for bundle in lib.make_bundles(len(traces), k):
        d = 0.0
        ok = True
        for i in bundle:
            a = traces[i]["arms"].get(arm, {}).get("recovered")
            b = traces[i]["arms"].get(other, {}).get("recovered")
            if a is None or b is None:
                ok = False
                break
            d += float(bool(a)) - float(bool(b))
        if not ok or d == 0:
            continue
        pos += int(d > 0)
        neg += int(d < 0)
    return pos, neg, pos + neg


def stage1_gate(traces: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    """Pre-registered S1-GO: at K=2 one build mode beats floor by margin, p<.05."""
    out: dict[str, Any] = {"margin": margin}
    go = False
    kf, nf = _rate(traces, "floor")
    for mode in ("adapter_a_k2", "adapter_b_k2"):
        km, nm = _rate(traces, mode)
        if not nm or not nf:
            out[mode] = {"passes": False, "reason": "arm missing"}
            continue
        a_only, b_only, _ = clamp._paired_discordants(traces, mode, "floor")
        p = clamp._two_sided_binom_p(a_only, b_only)
        delta = km / nm - kf / nf
        passes = p < 0.05 and delta >= margin
        out[mode] = {"rate": km / nm, "delta": delta, "p": p, "passes": passes}
        go = go or passes
    out["go"] = go
    return out


# ------------------------------------------------------------- model loading

def load_capacity_handles(cfg: Any, k_max: int) -> dict[str, Any]:
    """ModelWrapper + raw handles with the PEFT adapter at the campaign rank.

    Replicates ModelWrapper.from_config (wrapper.py) except LoraConfig.r —
    lora_alpha keeps the Sakana contract (alpha_peft = checkpoint_alpha *
    r_peft so PEFT's alpha/r quotient equals checkpoint_alpha at any rank).
    """
    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        load_hypernetwork,
    )
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyp = load_hypernetwork(
        HypernetworkConfig(
            checkpoint_path=cfg.checkpoint_path, model_config_name=cfg.model_id
        ),
        device=device,
    )
    hc = hyp.config
    rank = int(hc.lora_config.r)
    use_bias = bool(getattr(hc, "use_bias", False))
    bias_rank = rank if use_bias else 0
    alpha = float(getattr(hc.lora_config, "lora_alpha", rank * 2))
    r_camp = lib.campaign_rank(rank, bias_rank, k_max)
    raw = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    peft_model = get_peft_model(raw, LoraConfig(
        r=r_camp, lora_alpha=alpha * r_camp,
        target_modules=list(hc.lora_config.target_modules),
        lora_dropout=0.0, use_rslora=False,
    ))
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    return {
        "model": ModelWrapper(peft_model, tok, hyp, config=cfg),
        "hyp": hyp, "base": peft_model, "tok": tok,
        "li": [int(x) for x in hc.layer_indices],
        "target_modules": list(hc.lora_config.target_modules),
        "head_bias": hyp.get_head_bias() if use_bias else None,
        "ctx_rank": rank, "bias_rank": bias_rank, "r_camp": r_camp,
    }


def assemble_native_sd(h: dict[str, Any], text: str) -> dict[str, Any]:
    """Per-conditioning PEFT state dict at NATIVE (ctx+bias) rank.

    The probe's assembly path (_specificity_probe.py:277-283): activation
    extraction -> hypernet forward -> combine_lora -> _to_peft_state_dict.
    Caller must reset_adapter() first so activations come from the base model.
    """
    import torch  # noqa: PLC0415
    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415

    from rune.model.hypernetwork import (  # noqa: PLC0415
        _to_peft_state_dict,
        extract_activations_with_model,
    )

    feats, am = extract_activations_with_model(
        text=text, model=h["base"], tokenizer=h["tok"],
        layer_indices=h["li"], max_length=2048,
    )
    dev = next(h["hyp"].parameters()).device
    dt = next(h["hyp"].parameters()).dtype
    with torch.no_grad():
        ld, _ = h["hyp"].generate_weights(feats.to(device=dev, dtype=dt), am.to(dev), None)
    merged = combine_lora(ld, torch.tensor([1]), lora_bias=h["head_bias"])
    return _to_peft_state_dict(merged, h["li"], h["target_modules"])


# ------------------------------------------------------------------ the legs

async def run_capacity(h: dict[str, Any], rows: list[Any], args: Any) -> list[dict]:
    import torch  # noqa: PLC0415

    from rune.bench.repobench import render_episodic  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    model = h["model"]
    w = args.window
    ks = [int(x) for x in args.ks.split(",")]
    conds = [
        render_episodic(r, args.variant, anchor_chars=args.anchor)[:clamp._COND_CHAR_CAP]
        for r in rows
    ]
    traces: list[dict[str, Any]] = [
        {"task_id": r.task_id, "level": r.level, "gold_identifier": r.gold_identifier,
         "next_line": r.next_line, "arms": {}} for r in rows
    ]
    per_fact: dict[int, dict[str, Any]] = {}

    def fact_sd(i: int) -> dict[str, Any]:
        if i not in per_fact:
            model.reset_adapter()
            per_fact[i] = assemble_native_sd(h, conds[i])
        return per_fact[i]

    async def gen_scored(i: int, prompt: str) -> dict[str, Any]:
        torch.manual_seed(args.seed)
        return clamp._score(await clamp._gen_line(model, prompt, args.max_new), rows[i])

    floor_prompts = [
        model.clamp_to_window(
            f"# Current file:\n{clamp._prefix(r)}\n# Next line:", w
        ) for r in rows
    ]
    for i in range(len(rows)):
        model.reset_adapter()
        traces[i]["arms"]["floor"] = await gen_scored(i, floor_prompts[i])

    for k in ks:
        for bundle in lib.make_bundles(len(rows), k):
            joined = lib.multi_cond_text([conds[i] for i in bundle])
            overhead = model.count_tokens(
                f"{clamp._TAIL_HEADER}\n{joined}{clamp._CURSOR_MARKER}"
            )
            arm_t = f"tail_k{k}"
            for i in bundle:
                if overhead > w:
                    traces[i]["arms"][arm_t] = {
                        "pred": "", "recovered": False, "infeasible": True,
                        "cond_tokens": overhead,
                    }
                    continue
                model.reset_adapter()
                prompt, _ = clamp._assemble_tail_prompt(
                    model, clamp._prefix(rows[i]), joined, w
                )
                s = await gen_scored(i, prompt)
                s["cond_tokens"] = overhead
                traces[i]["arms"][arm_t] = s

            modes: list[tuple[str, dict[str, Any]]] = []
            if k == 1:
                modes.append(("adapter_k1", fact_sd(bundle[0])))
            else:
                if args.mode in ("both", "a"):
                    model.reset_adapter()
                    sd_a = assemble_native_sd(h, joined)
                    modes.append((f"adapter_a_k{k}", sd_a))
                if args.mode in ("both", "b"):
                    comp = lib.compose_rank_stacked(
                        [fact_sd(i) for i in bundle], ctx_rank=h["ctx_rank"]
                    )
                    modes.append((f"adapter_b_k{k}", comp))
            for label, sd in modes:
                model.reset_adapter()
                model.hotswap_adapter(
                    scale_lora_b(lib.pad_adapter_rank(sd, h["r_camp"]), args.scaling)
                )
                for i in bundle:
                    s = await gen_scored(i, floor_prompts[i])
                    s["k"] = k
                    traces[i]["arms"][label] = s
    return traces


async def run_sanity(rows: list[Any], args: Any, cfg: Any) -> list[dict]:
    """Native-rank leg through the ENGINE path; must reproduce C1 bit-exactly."""
    import torch  # noqa: PLC0415

    from rune.bench.repobench import render_episodic  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    model = ModelWrapper.from_config(cfg)
    traces: list[dict[str, Any]] = []
    for r in rows:
        floor_p = model.clamp_to_window(
            f"# Current file:\n{clamp._prefix(r)}\n# Next line:", args.window
        )
        rec: dict[str, Any] = {"task_id": r.task_id, "arms": {}}
        torch.manual_seed(args.seed)
        model.reset_adapter()
        rec["arms"]["floor"] = clamp._score(
            await clamp._gen_line(model, floor_p, args.max_new), r
        )
        cond = render_episodic(r, args.variant, anchor_chars=args.anchor)
        cond = cond[:clamp._COND_CHAR_CAP]
        model.reset_adapter()
        ar = model.generate_adapter(cond)
        torch.manual_seed(args.seed)
        model.hotswap_adapter(scale_lora_b(ar.state_dict, args.scaling))
        rec["arms"]["adapter_k1"] = clamp._score(
            await clamp._gen_line(model, floor_p, args.max_new), r
        )
        traces.append(rec)
    return traces


def compare_to_c1(traces: list[dict], c1_path: Path) -> dict[str, Any]:
    """Token-for-token prediction agreement vs the C1 trace artifact."""
    c1 = {t["task_id"]: t for t in json.loads(c1_path.read_text())}
    pairs = (("floor", "floor"), ("adapter_k1", "episodic_use"))
    out: dict[str, Any] = {}
    for ours, theirs in pairs:
        same = tot = 0
        for rec in traces:
            ref = c1.get(rec["task_id"], {}).get("arms", {}).get(theirs)
            arm = rec["arms"].get(ours)
            if not ref or not arm:
                continue
            tot += 1
            same += int(arm["pred"] == ref["pred"])
        out[f"match_{ours}"] = f"{same}/{tot}"
        out[f"match_{ours}_exact"] = same == tot
    return out


# ----------------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", choices=("sanity", "capacity"), default="capacity")
    ap.add_argument("--levels", default="8k,32k")
    ap.add_argument("--per-level", type=int, default=30)
    ap.add_argument("--offset", type=int, default=100)
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="use")
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--scaling", type=float, default=0.91)
    ap.add_argument("--ks", default=_KS_DEFAULT)
    ap.add_argument("--mode", choices=("both", "a", "b"), default="both")
    ap.add_argument("--margin", type=float, default=_MARGIN_DEFAULT)
    ap.add_argument("--experiment", default="issue52-c4")
    ap.add_argument("--out", default="/tmp/c4/capacity_traces.json")
    ap.add_argument("--c1-traces", default=None,
                    help="C1 run trace JSON for the sanity comparison")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute metrics + gate from an existing --out")
    ap.add_argument("--smoke", action="store_true",
                    help="first 8 rows, ks=1,2 (GPU plumbing check)")
    args = ap.parse_args()

    if args.stats_only:
        traces = json.loads(Path(args.out).read_text())
        m = capacity_metrics(traces)
        for key in sorted(m):
            print(f"{key} = {m[key]:.4f}")
        print(json.dumps(stage1_gate(traces, args.margin), indent=1))
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import asyncio  # noqa: PLC0415

    import mlflow  # noqa: PLC0415

    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    got = hashlib.sha256(Path(clamp.C3_CKPT).read_bytes()).hexdigest()
    if got != C3_SHA256:
        raise SystemExit(f"c3 ckpt sha {got} != pinned {C3_SHA256}")

    if args.smoke:
        args.ks = "1,2"
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    rows = clamp._load_stratified(levels, args.per_level, args.offset)
    if args.smoke:
        rows = rows[:8]
    cfg = load_rune_config(None).override(
        checkpoint_path=clamp.C3_CKPT, thinking_budget=0, seed=args.seed,
        max_tokens=args.max_new, temperature=0.0,
    )
    engine_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    ).stdout.strip()

    if args.leg == "sanity":
        traces = asyncio.run(run_sanity(rows, args, cfg))
    else:
        k_max = max(int(x) for x in args.ks.split(","))
        h = load_capacity_handles(cfg, k_max)
        traces = asyncio.run(run_capacity(h, rows, args))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(traces, indent=1))
    m = capacity_metrics(traces)
    gate = stage1_gate(traces, args.margin) if args.leg == "capacity" else {}
    anchor = (
        compare_to_c1(traces, Path(args.c1_traces))
        if args.leg == "sanity" and args.c1_traces else {}
    )

    configure_mlflow(args.experiment)
    run_name = f"c4-{args.leg}-W{args.window}-K{args.ks}-off{args.offset}-seed{args.seed}"
    params = {
        "task": "C4-stage1-I5", "leg": args.leg, "window": args.window,
        "ks": args.ks, "mode": args.mode, "margin": args.margin,
        "levels": args.levels, "per_level": args.per_level, "offset": args.offset,
        "seed": args.seed, "episodic_variant": args.variant,
        "episodic_anchor": args.anchor, "episodic_scaling": args.scaling,
        "max_new": args.max_new, "n_rows": len(rows),
        "checkpoint_sha256": got, "engine_commit": engine_commit,
        "c1_anchor_run": "f37374906c5f",
    }
    with tracked_run(run_name, params=params):
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in m.items()})
        mlflow.log_artifact(args.out)
        if gate:
            mlflow.log_dict(gate, "stage1_gate.json")
        if anchor:
            mlflow.log_dict(anchor, "c1_anchor.json")
    for key in sorted(m):
        print(f"{key} = {m[key]:.4f}")
    if anchor:
        print(json.dumps(anchor, indent=1))
    if gate:
        print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
