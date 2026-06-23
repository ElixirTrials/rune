"""RepoBench channel-strengthening probe (issue #52 long-context, cheap levers).

Probe 3 showed the adapter recovers a cross-file API the truncated prompt can't
(W=768), but weakly (1/6) — and at 8k levels the conditioning (~2.4k tok) exceeds
the hypernet's 2048 truncation, so a LATE gold snippet is evicted from the
adapter's view. This probe tests the cheap no-training levers, paired per row at
a fixed tight window:

  - nat@1.0     : natural snippet order, scaling 1.0 (Probe-3 control)
  - gf@1.0      : GOLD-FIRST order (gold def guaranteed within the 2048 budget)
  - gf@0.5      : gold-first, scaling 0.5
  - gf@1.0_cap4k: gold-first, conditioning budget lifted to 4096 (out-of-distribution)

Primary metric: gold-identifier recovery (did the cross-file API name appear).
Baselines per row: floor (no context, clamped) and a2_clamp (context in prompt,
clamped) — both at W. a2_full is the full-window ceiling.

Run: uv run --extra gpu python tools/_repobench_strengthen.py \
       --levels 8k,32k --per-level 6 --window 768 --out /tmp/rb_strength.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
_COND_CHAR_CAP = 16000

# (label, gold_first, scaling, max_length)
_VARIANTS = [
    ("nat@1.0", False, 1.0, 2048),
    ("gf@1.0", True, 1.0, 2048),
    ("gf@0.5", True, 0.5, 2048),
    ("gf@1.0_cap4k", True, 1.0, 4096),
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
    )

    gid = row.gold_identifier
    return {
        "pred": pred,
        "em": exact_match(pred, row.next_line),
        "es": round(edit_similarity(pred, row.next_line), 3),
        "recovered": bool(gold_id_recovery(pred, gid)) if gid else None,
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

    w = args.window
    traces: list[dict[str, Any]] = []
    for row in rows:
        prefix = _prefix(row)
        ctx = render_context_prompt(row)
        a2_full_prompt = (
            f"# Cross-file context:\n{ctx}\n\n# Current file:\n{prefix}\n# Next line:"
        )
        floor_p = model.clamp_to_window(f"# Current file:\n{prefix}\n# Next line:", w)
        a2c_p = model.clamp_to_window(a2_full_prompt, w)
        rec: dict[str, Any] = {
            "task_id": row.task_id,
            "repo": row.repo_name,
            "level": row.level,
            "token_num": row.token_num,
            "gold_identifier": row.gold_identifier,
            "gold_snippet_index": row.gold_snippet_index,
            "next_line": row.next_line,
            "n_context": len(row.context),
            "ctx_tokens": model.count_tokens(ctx),
            "arms": {},
        }
        try:
            torch.manual_seed(args.seed)
            model.reset_adapter()
            rec["arms"]["a2_full"] = _score(
                await _gen_line(model, a2_full_prompt, args.max_new), row
            )
            torch.manual_seed(args.seed)
            model.reset_adapter()
            floor = _score(await _gen_line(model, floor_p, args.max_new), row)
            rec["arms"]["floor"] = floor
            torch.manual_seed(args.seed)
            model.reset_adapter()
            rec["arms"]["a2_clamp"] = _score(
                await _gen_line(model, a2c_p, args.max_new), row
            )
            for label, gold_first, scaling, max_len in _VARIANTS:
                cond = render_xfile_adapter(row, "structured", gold_first=gold_first)[
                    :_COND_CHAR_CAP
                ]
                ar = model.generate_adapter(cond, max_length=max_len)
                torch.manual_seed(args.seed)
                model.hotswap_adapter(scale_lora_b(ar.state_dict, scaling))
                s = _score(await _gen_line(model, floor_p, args.max_new), row)
                s["cond_tokens"] = model.count_tokens(cond)
                s["win_vs_floor"] = s["es"] > floor["es"] + 1e-9
                s["recovers_beyond_prompt"] = bool(s["recovered"]) and not (
                    floor["recovered"] or rec["arms"]["a2_clamp"]["recovered"]
                )
                rec["arms"][label] = s
            model.reset_adapter()
        except Exception as e:  # noqa: BLE001 - probe: capture, continue
            rec["error"] = f"{type(e).__name__}: {e}"
        g = rec["arms"].get("gf@1.0", {})
        print(
            f"{row.task_id} [{row.level}] gold={row.gold_identifier!r} "
            f"gf@1.0_recov={g.get('recovered')} {rec.get('error', '')}",
            flush=True,
        )
        traces.append(rec)
    return traces


def _summary(traces: list[dict[str, Any]]) -> str:
    ok = [t for t in traces if "error" not in t]
    n = len(ok)
    lines = ["", f"=== STRENGTHEN SUMMARY (N={n}, window-clamped) ==="]
    by_level: dict[str, int] = {}
    for t in ok:
        by_level[t["level"]] = by_level.get(t["level"], 0) + 1
    lines.append(f"levels: {by_level}")
    lines.append("")
    lines.append(f"{'arm':<16} {'recovery':>9} {'es':>6} {'beyond_prompt':>14}")

    def rate(label: str, field: str) -> tuple[int, int]:
        vals = [t["arms"][label][field] for t in ok if label in t["arms"]]
        vals = [v for v in vals if v is not None]
        return sum(bool(v) for v in vals), len(vals)

    for label in ("a2_full", "floor", "a2_clamp", *[v[0] for v in _VARIANTS]):
        r, d = rate(label, "recovered")
        es = sum(t["arms"][label]["es"] for t in ok if label in t["arms"]) / (n or 1)
        if label in [v[0] for v in _VARIANTS]:
            bp, _ = rate(label, "recovers_beyond_prompt")
            beyond = f"{bp}/{n}"
        else:
            beyond = "-"
        lines.append(f"{label:<16} {r:>3}/{d:<5} {es:>6.3f} {beyond:>14}")
    lines.append("")
    lines.append("recovery = gold cross-file identifier appeared in the completion.")
    lines.append(
        "beyond_prompt = adapter recovered it where floor AND clamped-prompt did NOT."
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
    ap.add_argument("--levels", default="8k,32k")
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/rb_strength.json")
    args = ap.parse_args()

    import asyncio  # noqa: PLC0415

    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    rows = _load_stratified(levels, args.per_level)
    print(
        f"RepoBench rows: {len(rows)} (levels={levels} x {args.per_level}, W={args.window})",
        flush=True,
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
    print(_summary(traces), flush=True)
    print(f"\nwrote traces -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
