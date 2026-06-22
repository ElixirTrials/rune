"""RepoBench clamped-window benchmark — durable, scaled (issue #52 long-context).

The "scale up" runner for the cross-file-context-as-adapter experiment. Imposes a
small window budget W (constrained-hardware regime; Qwen3-4B's 262k window means
context otherwise always fits) and asks, at credible N with durable MLflow: when
the prompt can't hold the cross-file context, does the adapter (constant tiny
prompt) recover the cross-file API the truncated prompt cannot?

Arms (per row, gold-identifier recovery is primary):
- floor    : no context; prompt = clamp(prefix, W).
- a2_clamp : context in prompt, clamped to W (front-loaded context evicted).
- a2_full  : context in prompt at FULL window (ceiling; SKIPPED when the forward
             would be prohibitively large — that skip IS the cost argument).
- nat@1.0  : context in adapter, natural snippet order.
- gf@1.0   : context in adapter, gold-snippet-first (gold def within the 2048 budget).

Durable: MLflow params (config + engine_commit + checkpoint_sha + dataset id +
window + levels), metrics (per-arm recovery rate + beyond-prompt count + floor-vs
-adapter discordants), per-task JSONL artifact.

Run: uv run --extra gpu python tools/_repobench_clamp_run.py \
       --levels 8k,32k --per-level 30 --window 768 --experiment issue52-repobench-clamp \
       --out /tmp/rb_clamp_run.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
_COND_CHAR_CAP = 16000
_A2_FULL_MAX_TOKENS = 12000  # skip the full-context forward above this (OOM guard + cost arg)

# (label, gold_first, scaling, max_length) — adapter arms
_ADAPTER_ARMS = [
    ("nat@1.0", False, 1.0, 2048),
    ("gf@1.0", True, 1.0, 2048),
]

_SYSTEM = (
    "You are a code completion engine. Output ONLY the single next line of "
    "Python code that should follow the given file prefix. No explanation, no "
    "markdown fences, no blank lines."
)


def _first_code_line(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
    for line in t.splitlines():
        if line.strip() in ("", "```") or line.strip().startswith("```"):
            continue
        return line.rstrip()
    return ""


def _prefix(row: Any) -> str:
    return (row.import_statement + "\n\n" + row.cropped_code).strip()


async def _gen_line(model: Any, user: str, max_new: int) -> str:
    gen = await model.generate(
        prompt=user, system_prompt=_SYSTEM, output_schema=None, max_tokens=max_new,
        temperature=0.0, repetition_penalty=1.1, top_p=0.9, no_repeat_ngram_size=0,
        presence_penalty=0.0, thinking_budget=0,
    )
    return _first_code_line(gen.text)


def _score(pred: str, row: Any) -> dict[str, Any]:
    from rune.bench.identifier_match import (  # noqa: PLC0415
        edit_similarity,
        exact_match,
        gold_id_recovery,
    )

    gid = row.gold_identifier
    return {
        "pred": pred,
        "em": exact_match(pred, row.next_line),
        "es": round(edit_similarity(pred, row.next_line), 3),
        "recovered": bool(gold_id_recovery(pred, gid)) if gid else None,
    }


async def _run(model: Any, rows: list[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    from rune.bench.repobench import (  # noqa: PLC0415,E501
        render_context_prompt,
        render_xfile_adapter,
    )
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    w = args.window
    traces: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prefix = _prefix(row)
        ctx = render_context_prompt(row)
        ctx_tokens = model.count_tokens(ctx)
        a2_full_prompt = (
            f"# Cross-file context:\n{ctx}\n\n# Current file:\n{prefix}\n# Next line:"
        )
        floor_p = model.clamp_to_window(f"# Current file:\n{prefix}\n# Next line:", w)
        a2c_p = model.clamp_to_window(a2_full_prompt, w)
        rec: dict[str, Any] = {
            "task_id": row.task_id, "repo": row.repo_name, "level": row.level,
            "token_num": row.token_num, "gold_identifier": row.gold_identifier,
            "gold_snippet_index": row.gold_snippet_index, "next_line": row.next_line,
            "n_context": len(row.context), "ctx_tokens": ctx_tokens,
            "a2_full_prompt_tokens": model.count_tokens(a2_full_prompt),
            "arms": {},
        }
        try:
            torch.manual_seed(args.seed)
            model.reset_adapter()
            rec["arms"]["floor"] = _score(await _gen_line(model, floor_p, args.max_new), row)
            torch.manual_seed(args.seed)
            model.reset_adapter()
            rec["arms"]["a2_clamp"] = _score(await _gen_line(model, a2c_p, args.max_new), row)
            if ctx_tokens <= _A2_FULL_MAX_TOKENS:
                torch.manual_seed(args.seed)
                model.reset_adapter()
                rec["arms"]["a2_full"] = _score(await _gen_line(model, a2_full_prompt, args.max_new), row)
            else:
                rec["arms"]["a2_full"] = {"skipped": f"ctx_tokens>{_A2_FULL_MAX_TOKENS}"}
            for label, gold_first, scaling, max_len in _ADAPTER_ARMS:
                cond = render_xfile_adapter(row, "structured", gold_first=gold_first)[:_COND_CHAR_CAP]
                ar = model.generate_adapter(cond, max_length=max_len)
                torch.manual_seed(args.seed)
                model.hotswap_adapter(scale_lora_b(ar.state_dict, scaling))
                s = _score(await _gen_line(model, floor_p, args.max_new), row)
                s["recovers_beyond_prompt"] = bool(s["recovered"]) and not (
                    rec["arms"]["floor"]["recovered"] or rec["arms"]["a2_clamp"]["recovered"]
                )
                rec["arms"][label] = s
            model.reset_adapter()
        except Exception as e:  # noqa: BLE001 - capture per-row, keep the campaign alive
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        g = rec["arms"].get("gf@1.0", {})
        print(f"[{idx + 1}/{len(rows)}] {row.task_id} [{row.level}] gold={row.gold_identifier!r} "
              f"gf_recov={g.get('recovered')} {rec.get('error', '')}", flush=True)
        traces.append(rec)
    return traces


def _two_sided_binom_p(b: int, c: int) -> float:
    """Exact McNemar two-sided p for discordants (b, c) — no scipy dependency."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2 * tail)


def _metrics(traces: list[dict[str, Any]]) -> dict[str, float]:
    ok = [t for t in traces if "error" not in t]
    n = len(ok)
    out: dict[str, float] = {"n_ok": n, "n_total": len(traces)}

    def rate(label: str) -> tuple[int, int]:
        vals = [
            t["arms"][label]["recovered"]
            for t in ok
            if label in t["arms"] and "recovered" in t["arms"][label]
        ]
        vals = [v for v in vals if v is not None]
        return sum(bool(v) for v in vals), len(vals)

    for label in ("floor", "a2_clamp", "a2_full", "nat@1.0", "gf@1.0"):
        r, d = rate(label)
        out[f"recovery_{label}"] = r / d if d else 0.0
        out[f"recovery_{label}_n"] = r
        out[f"denom_{label}"] = d
    out["beyond_prompt_nat"] = sum(
        1 for t in ok if t["arms"].get("nat@1.0", {}).get("recovers_beyond_prompt")
    )
    out["beyond_prompt_gf"] = sum(
        1 for t in ok if t["arms"].get("gf@1.0", {}).get("recovers_beyond_prompt")
    )
    # McNemar floor vs best adapter (gf@1.0) on recovery
    b = sum(  # adapter recovers, floor does not
        1 for t in ok
        if t["arms"].get("gf@1.0", {}).get("recovered") and not t["arms"]["floor"]["recovered"]
    )
    c = sum(  # floor recovers, adapter does not
        1 for t in ok
        if t["arms"]["floor"].get("recovered") and not t["arms"].get("gf@1.0", {}).get("recovered")
    )
    out["mcnemar_adapter_only"] = b
    out["mcnemar_floor_only"] = c
    out["mcnemar_p"] = _two_sided_binom_p(b, c)
    return out


def _fmt_metrics(m: dict[str, float]) -> str:
    lines = ["", f"=== CLAMP RUN METRICS (N={int(m['n_ok'])}/{int(m['n_total'])}) ==="]
    lines.append(f"{'arm':<12}{'recovery':>12}")
    for label in ("floor", "a2_clamp", "a2_full", "nat@1.0", "gf@1.0"):
        r, d = int(m[f"recovery_{label}_n"]), int(m[f"denom_{label}"])
        lines.append(f"{label:<12}{r:>4}/{d:<4} = {m[f'recovery_{label}']:.3f}")
    lines.append("")
    lines.append(f"beyond-prompt (adapter recovers where floor AND clamped-prompt fail): "
                 f"nat={int(m['beyond_prompt_nat'])} gf={int(m['beyond_prompt_gf'])}")
    lines.append(f"McNemar floor vs gf@1.0: adapter_only={int(m['mcnemar_adapter_only'])} "
                 f"floor_only={int(m['mcnemar_floor_only'])} p={m['mcnemar_p']:.4f}")
    return "\n".join(lines)


def _load_stratified(levels: list[str], per_level: int) -> list[Any]:
    from rune.bench.repobench import load_repobench_rows  # noqa: PLC0415

    rows: list[Any] = []
    for lvl in levels:
        rows.extend(load_repobench_rows(limit=per_level, level=lvl))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="8k,32k")
    ap.add_argument("--per-level", type=int, default=30)
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment", default="issue52-repobench-clamp")
    ap.add_argument("--out", default="/tmp/rb_clamp_run.json")
    args = ap.parse_args()

    # OOM hardening: reduce CUDA fragmentation for the long-context forwards.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import asyncio  # noqa: PLC0415

    import mlflow  # noqa: PLC0415

    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    rows = _load_stratified(levels, args.per_level)
    print(f"RepoBench rows: {len(rows)} (levels={levels} x {args.per_level}, W={args.window})", flush=True)
    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT, thinking_budget=0, seed=args.seed,
        max_tokens=args.max_new, temperature=0.0,
    )
    model = ModelWrapper.from_config(cfg)

    ckpt_sha = hashlib.sha256(Path(C3_CKPT).read_bytes()).hexdigest()
    engine_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    ).stdout.strip()
    configure_mlflow(args.experiment)
    params = {
        **cfg.to_dict(),
        "benchmark": "repobench_v1.1_python",
        "split": "cross_file_first",
        "dataset_id": "tianyang/repobench_python_v1.1",
        "window": args.window,
        "levels": ",".join(levels),
        "per_level": args.per_level,
        "n_tasks": len(rows),
        "checkpoint_sha256": ckpt_sha,
        "engine_commit": engine_commit,
        "a2_full_max_tokens": _A2_FULL_MAX_TOKENS,
    }
    run_name = f"clamp-W{args.window}-{'_'.join(levels)}-n{len(rows)}-seed{args.seed}"
    with tracked_run(run_name, params=params):
        traces = asyncio.run(_run(model, rows, args))
        out_path = Path(args.out)
        out_path.write_text(json.dumps(traces, indent=1))
        mlflow.log_artifact(str(out_path))  # durable: per-task predictions + scores
        m = _metrics(traces)
        for k, v in m.items():
            # MLflow metric names forbid '@' (arm labels nat@1.0 / gf@1.0)
            mlflow.log_metric(k.replace("@", "_at_"), float(v))
    print(_fmt_metrics(m), flush=True)
    print(f"\nwrote traces -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
