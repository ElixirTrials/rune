"""RepoBench clamped-window probe — the JTBD#3 regime (issue #52 long-context).

Qwen3-4B has a 262k window, so RepoBench context (<=128k) ALWAYS fits the prompt
and the literal "context doesn't fit" conjecture is untestable at full window.
This probe IMPOSES a small window budget W (the constrained-hardware scenario
rune targets) and asks: when the prompt can no longer hold the cross-file
context, does delivering it via the adapter (constant tiny prompt) beat a
truncated prompt?

Arms (per row, escalate = best-of(floor, adapter), strictly no regression):
- floor      : no context; prompt = clamp(prefix, W).
- A2_full    : context in prompt at FULL window (the unconstrained ceiling).
- A2_clamp   : context in prompt, clamped to W (front-loaded context evicted).
- A3_clamp   : context in the ADAPTER; prompt = clamp(prefix, W); escalate/floor.

Run: uv run --extra gpu python tools/_repobench_clamp_probe.py \
       --levels 8k --per-level 6 --windows 768,1536 --out /tmp/rb_clamp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
_COND_CHAR_CAP = 12000  # ~adapter conditioning budget (hypernet caps tokens at 2048)
_FLOOR_FAIL_ES = 0.5  # es below this = floor "needs" the cross-file context

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


def _floor_prompt(prefix: str) -> str:
    return f"# Current file:\n{prefix}\n# Next line:"


def _ctx_prompt(ctx: str, prefix: str) -> str:
    return f"# Cross-file context:\n{ctx}\n\n# Current file:\n{prefix}\n# Next line:"


async def _gen_line(model: Any, user: str, max_new: int) -> str:
    gen = await model.generate(
        prompt=user,
        system_prompt=_SYSTEM,
        output_schema=None,
        max_tokens=max_new,
        temperature=0.0,
        repetition_penalty=1.1,
        top_p=0.9,
        no_repeat_ngram_size=0,
        presence_penalty=0.0,
        thinking_budget=0,
    )
    return _first_code_line(gen.text)


def _score(pred: str, row: Any) -> dict[str, Any]:
    from rune.bench.identifier_match import (  # noqa: PLC0415
        edit_similarity,
        exact_match,
        gold_id_recovery,
        identifier_f1,
    )

    gid = row.gold_identifier
    return {
        "pred": pred,
        "em": exact_match(pred, row.next_line),
        "es": round(edit_similarity(pred, row.next_line), 3),
        "id_f1": round(identifier_f1(pred, row.next_line), 3),
        "recovered": gold_id_recovery(pred, gid) if gid else None,
    }


async def _run(
    model: Any, rows: list[Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    from rune.bench.repobench import (  # noqa: PLC0415,E501
        render_context_prompt,
        render_xfile_adapter,
    )
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    traces: list[dict[str, Any]] = []
    for row in rows:
        prefix = _prefix(row)
        ctx = render_context_prompt(row)  # natural order
        cond = render_xfile_adapter(row, args.template)[:_COND_CHAR_CAP]
        a2_full_prompt = _ctx_prompt(ctx, prefix)
        rec: dict[str, Any] = {
            "task_id": row.task_id,
            "repo": row.repo_name,
            "level": row.level,
            "token_num": row.token_num,
            "gold_identifier": row.gold_identifier,
            "next_line": row.next_line,
            "n_context": len(row.context),
            "ctx_tokens": model.count_tokens(ctx),
            "cond_tokens": model.count_tokens(cond),
            "a2_full_prompt_tokens": model.count_tokens(a2_full_prompt),
            "windows": {},
        }
        try:
            # full-window context-in-prompt ceiling (window-independent)
            torch.manual_seed(args.seed)
            model.reset_adapter()
            rec["a2_full"] = _score(
                await _gen_line(model, a2_full_prompt, args.max_new), row
            )
            # adapter is window-independent too: build once, reuse per window
            ar = model.generate_adapter(cond)
            for w in windows:
                floor_p = model.clamp_to_window(_floor_prompt(prefix), w)
                a2c_p = model.clamp_to_window(a2_full_prompt, w)
                torch.manual_seed(args.seed)
                model.reset_adapter()
                floor = _score(await _gen_line(model, floor_p, args.max_new), row)
                torch.manual_seed(args.seed)
                model.reset_adapter()
                a2c = _score(await _gen_line(model, a2c_p, args.max_new), row)
                torch.manual_seed(args.seed)
                model.hotswap_adapter(scale_lora_b(ar.state_dict, args.scaling))
                a3 = _score(await _gen_line(model, floor_p, args.max_new), row)
                a3["escalate_es"] = max(floor["es"], a3["es"])
                a3["win_vs_floor"] = a3["es"] > floor["es"] + 1e-9
                rec["windows"][str(w)] = {
                    "floor_prompt_tokens": model.count_tokens(floor_p),
                    "a2clamp_prompt_tokens": model.count_tokens(a2c_p),
                    "ctx_survived_clamp": model.count_tokens(a2c_p)
                    > model.count_tokens(floor_p) + 2,
                    "floor": floor,
                    "a2_clamp": a2c,
                    "a3_clamp": a3,
                }
            model.reset_adapter()
        except Exception as e:  # noqa: BLE001 - probe: capture, don't abort the sweep
            rec["error"] = f"{type(e).__name__}: {e}"
        w0 = str(windows[0])
        fe = rec.get("windows", {}).get(w0, {}).get("floor", {}).get("es")
        print(
            f"{row.task_id} [{row.level}] floor_es@{w0}={fe} {rec.get('error', '')}",
            flush=True,
        )
        traces.append(rec)
    return traces


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _summary(traces: list[dict[str, Any]], args: argparse.Namespace) -> str:
    ok = [t for t in traces if "error" not in t]
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    lines = ["", f"=== CLAMP SUMMARY (N={len(ok)} ok / {len(traces)}) ==="]
    a2f = _mean([t["a2_full"]["es"] for t in ok])
    a2f_tok = _mean([t["a2_full_prompt_tokens"] for t in ok])
    lines.append(
        f"A2_full (ceiling, full window): es={a2f:.3f}  prompt_tok={a2f_tok:.0f}"
    )
    lines.append("")
    lines.append(
        f"{'window':>7} {'floor_es':>8} {'a2clamp_es':>10} {'a3clamp_es':>10} "
        f"{'a3_esc_es':>9} {'a3_wins':>7} {'ctx_survived':>12}"
    )
    for w in windows:
        cells = [t["windows"][w] for t in ok if w in t.get("windows", {})]
        if not cells:
            continue
        fe = _mean([c["floor"]["es"] for c in cells])
        a2c = _mean([c["a2_clamp"]["es"] for c in cells])
        a3 = _mean([c["a3_clamp"]["es"] for c in cells])
        esc = _mean([c["a3_clamp"]["escalate_es"] for c in cells])
        wins = sum(1 for c in cells if c["a3_clamp"]["win_vs_floor"])
        surv = sum(1 for c in cells if c["ctx_survived_clamp"])
        lines.append(
            f"{w:>7} {fe:>8.3f} {a2c:>10.3f} {a3:>10.3f} {esc:>9.3f} "
            f"{wins:>2}/{len(cells):<4} {surv:>3}/{len(cells):<4}"
        )

    # Floor-fail subset (where the cross-file context is genuinely needed)
    lines.append("")
    lines.append(f"--- floor-fail subset (floor es < {_FLOOR_FAIL_ES}) ---")
    for w in windows:
        cells = [
            t["windows"][w]
            for t in ok
            if w in t.get("windows", {})
            and t["windows"][w]["floor"]["es"] < _FLOOR_FAIL_ES
        ]
        if not cells:
            lines.append(f"{w:>7}  (none)")
            continue
        fe = _mean([c["floor"]["es"] for c in cells])
        a2c = _mean([c["a2_clamp"]["es"] for c in cells])
        a3 = _mean([c["a3_clamp"]["es"] for c in cells])
        a2full = _mean(
            [
                t["a2_full"]["es"]
                for t in ok
                if w in t.get("windows", {})
                and t["windows"][w]["floor"]["es"] < _FLOOR_FAIL_ES
            ]
        )
        wins = sum(1 for c in cells if c["a3_clamp"]["win_vs_floor"])
        lines.append(
            f"{w:>7} n={len(cells)}  floor={fe:.3f}  a2_clamp={a2c:.3f}  "
            f"a3_clamp={a3:.3f}  a3_wins={wins}/{len(cells)}  (a2_full={a2full:.3f})"
        )
    return "\n".join(lines)


def _load_stratified(levels: list[str], per_level: int) -> list[Any]:
    from rune.bench.repobench import load_repobench_rows  # noqa: PLC0415

    rows: list[Any] = []
    for lvl in levels:
        rows.extend(load_repobench_rows(limit=per_level, level=lvl))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="8k", help="comma-separated RepoBench levels")
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument(
        "--windows", default="768,1536", help="comma-separated token budgets"
    )
    ap.add_argument(
        "--template", default="structured", choices=["raw", "structured", "training"]
    )
    ap.add_argument("--scaling", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/rb_clamp.json")
    args = ap.parse_args()

    import asyncio  # noqa: PLC0415

    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    rows = _load_stratified(levels, args.per_level)
    print(
        f"RepoBench rows: {len(rows)} (levels={levels} x {args.per_level})", flush=True
    )
    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT,
        thinking_budget=0,
        seed=args.seed,
        max_tokens=args.max_new,
        temperature=0.0,
    )
    model = ModelWrapper.from_config(cfg)
    traces = asyncio.run(_run(model, rows, args))
    Path(args.out).write_text(json.dumps(traces, indent=1))
    print(_summary(traces, args), flush=True)
    print(f"\nwrote traces -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
