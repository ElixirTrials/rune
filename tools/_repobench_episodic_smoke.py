"""Episodic-template smoke test (issue #52 long-context, corrected adapter template).

The N=60 clamp run fed the adapter a multi-file repo DUMP (render_xfile_adapter
'structured'), which the hypernet's 2048-token cap shredded — not an episodic,
per-task conditioning. This smoke tests the corrected template on the tasks where
a2_full (full context in the prompt) FAILED to recover the gold API: does a focused
episodic adapter ("you must call X, here is X") recover where even the full-context
prompt could not?

Arms (per task, gold-identifier recovery): floor (no context), a2_full (full ctx in
prompt, the failed ceiling), dump_gf (OLD multi-file dump in adapter), episodic_gold
(NEW: gold def only, training surface), episodic_sig (NEW: gold signature only).
Adapter arms use the clamped tiny prompt; only the adapter conditioning differs.

Run: uv run --extra gpu python tools/_repobench_episodic_smoke.py \
       --task-ids cross_file_first/2000,... --level 8k --window 768
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
_COND_CHAR_CAP = 16000

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
        gold_id_recovery,
    )

    gid = row.gold_identifier
    return {
        "pred": pred,
        "es": round(edit_similarity(pred, row.next_line), 3),
        "recovered": bool(gold_id_recovery(pred, gid)) if gid else None,
    }


async def _run(
    model: Any, rows: list[Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    from rune.bench.repobench import (  # noqa: PLC0415
        render_context_prompt,
        render_episodic_adapter,
        render_xfile_adapter,
    )
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    w = args.window
    # (label, conditioning-render fn)
    adapter_arms = [
        ("dump_gf", lambda r: render_xfile_adapter(r, "structured", gold_first=True)),
        ("episodic_gold", render_episodic_adapter),
        ("episodic_sig", lambda r: render_episodic_adapter(r, signature_only=True)),
    ]
    traces: list[dict[str, Any]] = []
    for row in rows:
        prefix = _prefix(row)
        ctx = render_context_prompt(row)
        a2_full_prompt = (
            f"# Cross-file context:\n{ctx}\n\n# Current file:\n{prefix}\n# Next line:"
        )
        floor_p = model.clamp_to_window(f"# Current file:\n{prefix}\n# Next line:", w)
        rec: dict[str, Any] = {
            "task_id": row.task_id,
            "level": row.level,
            "gold_identifier": row.gold_identifier,
            "next_line": row.next_line,
            "arms": {},
        }
        try:
            torch.manual_seed(args.seed)
            model.reset_adapter()
            rec["arms"]["floor"] = _score(
                await _gen_line(model, floor_p, args.max_new), row
            )
            torch.manual_seed(args.seed)
            model.reset_adapter()
            rec["arms"]["a2_full"] = _score(
                await _gen_line(model, a2_full_prompt, args.max_new), row
            )
            for label, render in adapter_arms:
                cond = render(row)[:_COND_CHAR_CAP]
                ar = model.generate_adapter(cond)
                torch.manual_seed(args.seed)
                model.hotswap_adapter(scale_lora_b(ar.state_dict, 1.0))
                s = _score(await _gen_line(model, floor_p, args.max_new), row)
                s["cond_tokens"] = model.count_tokens(cond)
                rec["arms"][label] = s
            model.reset_adapter()
        except Exception as e:  # noqa: BLE001 - smoke: capture, continue
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"--- {row.task_id} gold={row.gold_identifier!r} ---", flush=True)
        for label in ("floor", "a2_full", "dump_gf", "episodic_gold", "episodic_sig"):
            a = rec["arms"].get(label, {})
            if "recovered" in a:
                flag = "RECOVER" if a["recovered"] else "  miss "
                ct = f" cond={a['cond_tokens']}t" if "cond_tokens" in a else ""
                print(
                    f"   {label:<14} [{flag}] es={a['es']:<5}{ct}  {a['pred']!r}",
                    flush=True,
                )
        traces.append(rec)
    return traces


def _summary(traces: list[dict[str, Any]]) -> str:
    ok = [t for t in traces if "error" not in t]
    n = len(ok)
    lines = ["", f"=== EPISODIC SMOKE SUMMARY (N={n}, tasks where a2_full FAILED) ==="]
    for label in ("floor", "a2_full", "dump_gf", "episodic_gold", "episodic_sig"):
        rec = sum(1 for t in ok if t["arms"].get(label, {}).get("recovered"))
        es = sum(t["arms"][label]["es"] for t in ok if label in t["arms"]) / (n or 1)
        lines.append(f"  {label:<14} recovery {rec}/{n}   mean_es={es:.3f}")
    lines.append("")
    lines.append(
        "(a2_full=0 by construction — these are its failures; any episodic recovery is NET-NEW.)"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task-ids", required=True, help="comma-separated cross_file_first/<i>"
    )
    ap.add_argument("--level", default="8k")
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/rb_episodic_smoke.json")
    args = ap.parse_args()

    import asyncio  # noqa: PLC0415

    from rune.bench.repobench import load_repobench_rows  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    want = {x.strip() for x in args.task_ids.split(",") if x.strip()}
    rows = [r for r in load_repobench_rows(level=args.level) if r.task_id in want]
    print(
        f"smoke rows: {len(rows)}/{len(want)} requested (level={args.level}, W={args.window})",
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
