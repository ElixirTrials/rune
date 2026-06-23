"""RepoBench cross-file completion probe — escalate + template/scaling sweep.

Issue #52 long-context probe v2. Fixes three issues found in v1:
- ESCALATE (not replace): the adapter is scored as best-of(zero-shot floor,
  adapter candidate), mirroring the rune runner's escalation + keep-best, so the
  adapter can only ADD wins over base — strictly no regression (PR #57 framing).
- Metric: bare-identifier recovery is dead (gold name is in the import line for
  ~99.9% of rows), so we lead with edit-similarity + exact-match (RepoBench
  native) and keep identifier-F1 secondary.
- Sampling: stratify across RepoBench `level` buckets (v1 sampled only 2k).

Sweeps the no-training levers: render template (structured | training-format)
x adapter scaling. A3 drives ModelWrapper directly (generate_adapter -> hotswap
-> generate); the zero-shot floor and the context-in-prompt baseline are
adapter-off generations.

Run: uv run --extra gpu python tools/_repobench_probe.py \
       --levels 2k,8k --per-level 4 --out /tmp/rb_probe2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
_COND_CHAR_CAP = 8000  # cap conditioning text fed to the hypernet (probe-bounded)

_SYSTEM = (
    "You are a code completion engine. Output ONLY the single next line of "
    "Python code that should follow the given file prefix. No explanation, no "
    "markdown fences, no blank lines."
)


def _first_code_line(text: str) -> str:
    """First non-blank code line of a (possibly fenced/chatty) generation."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
    for line in t.splitlines():
        if line.strip() in ("", "```") or line.strip().startswith("```"):
            continue
        return line.rstrip()
    return ""


def _user_prompt(row: Any, *, with_context: bool) -> str:
    from rune.bench.repobench import render_context_prompt  # noqa: PLC0415

    file_prefix = (row.import_statement + "\n\n" + row.cropped_code).strip()
    if with_context:
        ctx = render_context_prompt(row)
        return (
            f"# Cross-file context:\n{ctx}\n\n"
            f"# Current file:\n{file_prefix}\n# Next line:"
        )
    return f"# Current file:\n{file_prefix}\n# Next line:"


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
        identifier_f1,
    )

    return {
        "pred": pred,
        "em": exact_match(pred, row.next_line),
        "es": round(edit_similarity(pred, row.next_line), 3),
        "id_f1": round(identifier_f1(pred, row.next_line), 3),
    }


async def _run(
    model: Any, rows: list[Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    from rune.bench.repobench import render_xfile_adapter  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    scalings = [float(s) for s in args.scalings.split(",") if s.strip()]
    user_nc_key, user_ctx_key = "no_context", "context_in_prompt"
    traces: list[dict[str, Any]] = []
    for row in rows:
        user_nc = _user_prompt(row, with_context=False)
        user_ctx = _user_prompt(row, with_context=True)
        rec: dict[str, Any] = {
            "task_id": row.task_id,
            "repo": row.repo_name,
            "level": row.level,
            "token_num": row.token_num,
            "gold_identifier": row.gold_identifier,
            "next_line": row.next_line,
            "n_context": len(row.context),
            "prompt_tokens_nc": model.count_tokens(user_nc),
            "prompt_tokens_ctx": model.count_tokens(user_ctx),
            "arms": {},
        }
        try:
            torch.manual_seed(args.seed)
            model.reset_adapter()
            zs = _score(
                await _gen_line(model, user_nc, args.max_new), row
            )  # floor (A1)
            rec["arms"][user_nc_key] = zs
            model.reset_adapter()
            rec["arms"][user_ctx_key] = _score(
                await _gen_line(model, user_ctx, args.max_new), row
            )  # context-in-prompt baseline (A2)
            for tmpl in templates:
                cond = render_xfile_adapter(row, tmpl)[:_COND_CHAR_CAP]
                ar = model.generate_adapter(cond)
                cond_tokens = model.count_tokens(cond)
                for sc in scalings:
                    torch.manual_seed(args.seed)
                    model.hotswap_adapter(scale_lora_b(ar.state_dict, sc))
                    s = _score(await _gen_line(model, user_nc, args.max_new), row)
                    s["cond_tokens"] = cond_tokens
                    s["escalate_es"] = max(zs["es"], s["es"])  # keep-best floor
                    s["escalate_em"] = zs["em"] or s["em"]
                    s["win_es"] = s["es"] > zs["es"] + 1e-9  # adapter strictly adds
                    rec["arms"][f"{tmpl}@{sc:g}"] = s
            model.reset_adapter()
        except Exception as e:  # noqa: BLE001 - probe: capture, don't abort the sweep
            rec["error"] = f"{type(e).__name__}: {e}"
        ze = rec["arms"].get(user_nc_key, {}).get("es")
        print(
            f"{row.task_id} [{row.level}] zs_es={ze} {rec.get('error', '')}", flush=True
        )
        traces.append(rec)
    return traces


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _summary(traces: list[dict[str, Any]], args: argparse.Namespace) -> str:
    ok = [t for t in traces if "error" not in t]
    lines = ["", f"=== SUMMARY (N={len(ok)} ok / {len(traces)}) ==="]
    by_level: dict[str, int] = {}
    for t in ok:
        by_level[t["level"]] = by_level.get(t["level"], 0) + 1
    lines.append(f"levels: {by_level}")

    def col(key: str, metric: str) -> list[float]:
        return [t["arms"][key][metric] for t in ok if key in t["arms"]]

    lines.append("")
    lines.append(
        f"{'arm':<22} {'es':>6} {'em':>5} {'esc_es':>7} {'wins':>5} {'cond_tok':>9}"
    )
    for base in ("no_context", "context_in_prompt"):
        es, em = _mean(col(base, "es")), _mean(col(base, "em"))
        lines.append(f"{base:<22} {es:>6.3f} {em:>5.2f} {'-':>7} {'-':>5} {'-':>9}")

    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    scalings = [float(s) for s in args.scalings.split(",") if s.strip()]
    for tmpl in templates:
        for sc in scalings:
            key = f"{tmpl}@{sc:g}"
            cells = [t["arms"][key] for t in ok if key in t["arms"]]
            if not cells:
                continue
            es = _mean([c["es"] for c in cells])
            em = _mean([c["em"] for c in cells])
            esc = _mean([c["escalate_es"] for c in cells])
            wins = sum(1 for c in cells if c["win_es"])
            ct = _mean([c["cond_tokens"] for c in cells])
            lines.append(
                f"{key:<22} {es:>6.3f} {em:>5.2f} {esc:>7.3f} "
                f"{wins:>2}/{len(cells):<2} {ct:>9.0f}"
            )

    # Envelope: per task, best adapter es over ALL configs, then escalate vs floor.
    env_wins, env_esc, env_floor = 0, [], []
    for t in ok:
        zs = t["arms"]["no_context"]["es"]
        adapter_keys = [k for k in t["arms"] if "@" in k]
        best = max((t["arms"][k]["es"] for k in adapter_keys), default=zs)
        env_esc.append(max(zs, best))
        env_floor.append(zs)
        if best > zs + 1e-9:
            env_wins += 1
    lines.append("")
    lines.append(
        f"ENVELOPE (best adapter/task): floor_es={_mean(env_floor):.3f} "
        f"escalate_es={_mean(env_esc):.3f} adapter_adds_wins={env_wins}/{len(ok)}"
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
    ap.add_argument(
        "--levels", default="2k,8k", help="comma-separated RepoBench levels"
    )
    ap.add_argument("--per-level", type=int, default=4)
    ap.add_argument("--templates", default="structured,training")
    ap.add_argument("--scalings", default="1.0,0.5,0.25")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/rb_probe2.json")
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
