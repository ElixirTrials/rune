"""RepoBench clamped-window benchmark — durable, scaled (issue #52 long-context).

The "scale up" runner for the cross-file-context-as-adapter experiment. Imposes a
small window budget W (constrained-hardware regime; Qwen3-4B's 262k window means
context otherwise always fits) and asks, at credible N with durable MLflow: when
the prompt can't hold the cross-file context, does the adapter (constant tiny
prompt) recover the cross-file API the truncated prompt cannot?

Arms (per row, gold-identifier recovery is primary):
- floor        : no context; prompt = clamp(prefix, W).
- a2_clamp     : context in prompt, clamped to W (front-loaded context evicted).
- a2_full      : context in prompt at FULL window (ceiling; SKIPPED when the forward
                 would be prohibitively large — that skip IS the cost argument).
- episodic_use : context in adapter via the EPISODIC per-task template (name the one
                 cross-file API the task must call) — the template-HPO winner
                 (variant=use, anchor=0, scaling=0.91); CLI-overridable.
- dump_gf      : context in adapter via the OLD multi-file dump (regression reference).

Pre-registered C1 arms (publication_task_plan.md C1.1-C1.3; extend-only — the five
arms above keep their exact behavior/labels for prior-run comparability):
- a2_tail        : the IDENTICAL episodic conditioning string placed at the prompt
                   TAIL, adjacent to the cursor, prefix clamped so total <= W
                   (the honest in-prompt channel; remediation plan 1a).
- a2_tail_filler : same construction, conditioning replaced by NEUTRAL filler
                   token-matched to the conditioning length — isolates the
                   pointer's marginal contribution from token displacement.
- swap           : adapter generated from the episodic conditioning with the gold
                   identifier renamed to a different row's gold (HANDOFF Stats C7 /
                   design-spec s8 "wrong task's symbol"; donors sharing the gold's
                   surface form are inadmissible); scored for recovery of the
                   ORIGINAL gold on the floor prompt (frequency-confound control).

Durable: MLflow params (config + engine_commit + checkpoint_sha + dataset id +
window + levels), metrics (per-arm recovery rate + Wilson 95% CIs + beyond-prompt
count + floor-vs-adapter discordants + pre-registered paired McNemars +
attributable fraction (e-s)/(e-f)), per-task JSONL artifact.

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
import re
import subprocess
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
_COND_CHAR_CAP = 16000
_A2_FULL_MAX_TOKENS = 12000  # skip the full-context forward above this (OOM guard + cost arg)

# Best episodic config from the template HPO (issue52-repobench-template-hpo,
# held-out 4/10 vs floor 1/10, strict superset). CLI-overridable.
_EPISODIC_VARIANT = "use"
_EPISODIC_ANCHOR = 0
_EPISODIC_SCALING = 0.91
_ADAPTER_LABELS = ("episodic_use", "dump_gf")  # primary + regression reference

# Original five arms (behavior/labels/metric names frozen for prior-run
# comparability) + the pre-registered C1 arms (extend-only).
_PRIMARY_ARMS = ("floor", "a2_clamp", "a2_full", "episodic_use", "dump_gf")
_C1_ARMS = ("a2_tail", "a2_tail_filler", "swap")

# Pre-registered paired comparisons (remediation plan 1a/1b gates). The
# original floor-vs-episodic McNemar keeps its legacy metric names.
_MCNEMAR_PAIRS = (
    ("episodic_use", "a2_tail"),
    ("a2_tail", "a2_tail_filler"),
    ("swap", "floor"),
    ("swap", "episodic_use"),
)

# Neutral prose for the a2_tail_filler control: no task identifiers, no
# code-like tokens. Words colliding with the row's identifiers are filtered
# per-row before use.
_FILLER_WORDS = (
    "meanwhile", "quiet", "afternoon", "light", "settled", "gently", "across",
    "distant", "hills", "slow", "clouds", "drifted", "toward", "the", "horizon",
    "soft", "grass", "swayed", "beneath", "a", "mild", "breeze", "over", "meadow",
)

_TAIL_HEADER = "# Current file:\n"
_CURSOR_MARKER = "\n# Next line:"


def _assemble_tail_prompt(
    model: Any, prefix: str, cond: str, window: int
) -> tuple[str, str]:
    """Prompt ending with ``cond`` immediately before the cursor marker, <= window.

    The current-file prefix is tail-clamped so the TOTAL prompt fits the window
    budget — the conditioning displaces near-cursor code rather than extending
    the budget (remediation plan 1a: within-budget trade). Returns
    ``(prompt, clamped_prefix)`` so the caller can record the realized prefix
    token count (matched cursor-code-length comparison vs floor).
    """
    tail = f"\n{cond}{_CURSOR_MARKER}"
    budget = max(window - model.count_tokens(_TAIL_HEADER + tail), 0)
    clamped = model.clamp_to_window(prefix, budget)
    prompt = f"{_TAIL_HEADER}{clamped}{tail}"
    # Tokenization is not additive across the joins; shrink until the total fits.
    while budget > 0 and (over := model.count_tokens(prompt) - window) > 0:
        budget = max(budget - over, 0)
        clamped = model.clamp_to_window(prefix, budget)
        prompt = f"{_TAIL_HEADER}{clamped}{tail}"
    return prompt, clamped


def _neutral_filler(model: Any, target_tokens: int, forbidden: set[str]) -> str:
    """Neutral filler matched to ``target_tokens`` via ``model.count_tokens``.

    Carries no task information: words colliding (case-insensitively) with
    ``forbidden`` (the row's identifiers) are dropped; the rest is plain prose.
    Grows word-by-word to the target, trims overshoot, then closes any
    sub-word-size gap greedily so the token count matches exactly whenever a
    single-token word exists.
    """
    bad = {w.lower() for w in forbidden}
    words = [w for w in _FILLER_WORDS if w.lower() not in bad]
    if not words or target_tokens <= 0:
        return ""
    parts: list[str] = []
    i = 0
    while model.count_tokens(" ".join(parts)) < target_tokens and i < 8 * target_tokens:
        parts.append(words[i % len(words)])
        i += 1
    while parts and model.count_tokens(" ".join(parts)) > target_tokens:
        parts.pop()
    for _ in range(target_tokens):
        if model.count_tokens(" ".join(parts)) >= target_tokens:
            break
        nxt = next(
            (w for w in words if model.count_tokens(" ".join([*parts, w])) <= target_tokens),
            None,
        )
        if nxt is None:
            break
        parts.append(nxt)
    return " ".join(parts)


def _swap_conditioning(cond: str, gold: str, replacement: str) -> tuple[str, int]:
    """Replace ALL whole-token occurrences of ``gold`` in ``cond``.

    Returns ``(swapped_text, n_replaced)``; ``n_replaced == 0`` means the gold
    identifier does not occur in the conditioning and the row is
    swap-inapplicable (guard — never silently run an unswapped conditioning).
    """
    if not gold:
        return cond, 0
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(gold)}(?![A-Za-z0-9_])")
    return pat.subn(replacement, cond)


def _pick_swap_identifier(idx: int, golds: list[str]) -> str | None:
    """Donor identifier for row ``idx``: the next admissible gold (cyclic).

    HANDOFF Stats C7: condition on the *wrong* task's symbol — a plausible
    identifier drawn from the same benchmark, deterministic given row order.
    Donors sharing the gold's surface form are inadmissible (adjacent rows are
    same-repo, so golds share naming families): a donor containing the gold
    (gold 'Config' -> donor 'ConfigLoader') keeps the gold's characters in the
    swapped conditioning, and a donor contained in the gold (gold
    'ConfigLoader' -> donor 'Config') can be autoregressively extended back
    into it — either leak primes recovery of the original gold and biases the
    swap gate toward a spurious 'keystone compromised' outcome.
    """
    gold = golds[idx]
    if not gold:
        return None
    gold_l = gold.lower()
    for off in range(1, len(golds)):
        cand = golds[(idx + off) % len(golds)]
        if not cand or cand == gold:
            continue
        cand_l = cand.lower()
        if gold_l in cand_l or cand_l in gold_l:
            continue
        return cand
    return None


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% score interval for k successes in n trials — no scipy dependency."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    # boundary cases are exact analytically; avoid float round-off drift
    lo = max(0.0, center - half) if k > 0 else 0.0
    hi = min(1.0, center + half) if k < n else 1.0
    return (lo, hi)


def _paired_discordants(
    traces: list[dict[str, Any]], arm_a: str, arm_b: str
) -> tuple[int, int, int]:
    """McNemar counts over rows where BOTH arms have non-null 'recovered'.

    Returns ``(a_only, b_only, n_pairs)``; skipped arms and null-gold rows are
    excluded from the pair, matching the pre-registration.
    """
    a_only = b_only = n = 0
    for t in traces:
        ra = t["arms"].get(arm_a, {}).get("recovered")
        rb = t["arms"].get(arm_b, {}).get("recovered")
        if ra is None or rb is None:
            continue
        n += 1
        a_only += int(bool(ra) and not rb)
        b_only += int(bool(rb) and not ra)
    return a_only, b_only, n

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

    from rune.bench.identifier_match import (  # noqa: PLC0415
        extract_identifiers,
        gold_id_recovery,
    )
    from rune.bench.repobench import (  # noqa: PLC0415,E501
        render_context_prompt,
        render_episodic,
        render_xfile_adapter,
    )
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415

    w = args.window
    golds = [r.gold_identifier for r in rows]  # swap-donor pool (wrong-task symbols)
    # (label, conditioning text, scaling): the validated episodic arm + the dump
    # regression reference. Episodic conditioning names the ONE cross-file API.
    def adapter_arms(row: Any) -> list[tuple[str, str, float]]:
        return [
            ("episodic_use",
             render_episodic(row, args.variant, anchor_chars=args.anchor), args.scaling),
            ("dump_gf",
             render_xfile_adapter(row, "structured", gold_first=True), 1.0),
        ]
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
            for label, cond_text, scaling in adapter_arms(row):
                cond = cond_text[:_COND_CHAR_CAP]
                ar = model.generate_adapter(cond)
                torch.manual_seed(args.seed)
                model.hotswap_adapter(scale_lora_b(ar.state_dict, scaling))
                s = _score(await _gen_line(model, floor_p, args.max_new), row)
                s["cond_tokens"] = model.count_tokens(cond)
                s["recovers_beyond_prompt"] = bool(s["recovered"]) and not (
                    rec["arms"]["floor"]["recovered"] or rec["arms"]["a2_clamp"]["recovered"]
                )
                rec["arms"][label] = s

            # --- pre-registered C1 arms (extend-only; the five arms above are
            # untouched for prior-run comparability) ---
            rec["floor_prompt_tokens"] = model.count_tokens(floor_p)
            # a2_tail: the IDENTICAL conditioning string episodic_use receives,
            # at the prompt tail, within the same window budget (channel test).
            cond_e = render_episodic(row, args.variant, anchor_chars=args.anchor)[:_COND_CHAR_CAP]
            tail_overhead = model.count_tokens(f"{_TAIL_HEADER}\n{cond_e}{_CURSOR_MARKER}")
            if tail_overhead > w:
                # The within-budget trade (remediation plan 1a) is undefined when
                # the conditioning alone exceeds W: skip rather than overflow.
                skip = {"skipped": f"tail_overhead_tokens>{w}",
                        "cond_tokens": model.count_tokens(cond_e)}
                rec["arms"]["a2_tail"] = dict(skip)
                rec["arms"]["a2_tail_filler"] = dict(skip)
            else:
                tail_p, tail_prefix = _assemble_tail_prompt(model, prefix, cond_e, w)
                torch.manual_seed(args.seed)
                model.reset_adapter()
                s = _score(await _gen_line(model, tail_p, args.max_new), row)
                s["cond_tokens"] = model.count_tokens(cond_e)
                s["prefix_tokens"] = model.count_tokens(tail_prefix)
                s["prompt_tokens"] = model.count_tokens(tail_p)
                rec["arms"]["a2_tail"] = s

                # a2_tail_filler: identical construction, neutral filler token-
                # matched to the conditioning — isolates the pointer from token
                # displacement.
                forbidden = {row.gold_identifier, *extract_identifiers(cond_e)}
                filler = _neutral_filler(model, model.count_tokens(cond_e), forbidden)
                fill_p, fill_prefix = _assemble_tail_prompt(model, prefix, filler, w)
                torch.manual_seed(args.seed)
                model.reset_adapter()
                s = _score(await _gen_line(model, fill_p, args.max_new), row)
                s["filler_tokens"] = model.count_tokens(filler)
                s["prefix_tokens"] = model.count_tokens(fill_prefix)
                s["prompt_tokens"] = model.count_tokens(fill_p)
                rec["arms"]["a2_tail_filler"] = s

            # swap: adapter from the conditioning with the gold identifier renamed
            # to a different row's gold; scored for recovery of the ORIGINAL gold
            # on the same floor prompt (frequency/output-bias control, C7).
            gold = row.gold_identifier
            donor = _pick_swap_identifier(idx, golds)
            if not gold or donor is None:
                rec["arms"]["swap"] = {
                    "skipped": "swap-inapplicable: no gold identifier or no donor"
                }
            else:
                swapped_cond, n_swaps = _swap_conditioning(cond_e, gold, donor)
                if n_swaps == 0:
                    rec["arms"]["swap"] = {
                        "skipped": "swap-inapplicable: gold absent from conditioning"
                    }
                else:
                    ar = model.generate_adapter(swapped_cond)
                    torch.manual_seed(args.seed)
                    model.hotswap_adapter(scale_lora_b(ar.state_dict, args.scaling))
                    pred = await _gen_line(model, floor_p, args.max_new)
                    s = _score(pred, row)  # primary: recovery of the ORIGINAL gold
                    s["cond_tokens"] = model.count_tokens(swapped_cond)
                    s["swap_identifier"] = donor
                    s["swap_occurrences"] = n_swaps
                    # PR #57 s8 content-vs-pointer signal: did it track the rename?
                    s["swapped_recovered"] = bool(gold_id_recovery(pred, donor))
                    rec["arms"]["swap"] = s
            model.reset_adapter()
        except Exception as e:  # noqa: BLE001 - capture per-row, keep the campaign alive
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        g = rec["arms"].get("episodic_use", {})
        print(f"[{idx + 1}/{len(rows)}] {row.task_id} [{row.level}] gold={row.gold_identifier!r} "
              f"episodic_recov={g.get('recovered')} {rec.get('error', '')}", flush=True)
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

    for label in (*_PRIMARY_ARMS, *_C1_ARMS):
        r, d = rate(label)
        out[f"recovery_{label}"] = r / d if d else 0.0
        out[f"recovery_{label}_n"] = r
        out[f"denom_{label}"] = d
        lo, hi = _wilson_ci(r, d)
        out[f"recovery_{label}_wilson_lo"] = lo
        out[f"recovery_{label}_wilson_hi"] = hi
    out["beyond_prompt_episodic"] = sum(
        1 for t in ok if t["arms"].get("episodic_use", {}).get("recovers_beyond_prompt")
    )
    out["beyond_prompt_dump"] = sum(
        1 for t in ok if t["arms"].get("dump_gf", {}).get("recovers_beyond_prompt")
    )
    # McNemar floor vs best adapter (episodic_use) on recovery
    b = sum(  # adapter recovers, floor does not
        1 for t in ok
        if t["arms"].get("episodic_use", {}).get("recovered") and not t["arms"]["floor"]["recovered"]
    )
    c = sum(  # floor recovers, adapter does not
        1 for t in ok
        if t["arms"]["floor"].get("recovered") and not t["arms"].get("episodic_use", {}).get("recovered")
    )
    out["mcnemar_adapter_only"] = b
    out["mcnemar_floor_only"] = c
    out["mcnemar_p"] = _two_sided_binom_p(b, c)
    # Pre-registered paired McNemars (C1 gates), over rows where both arms scored.
    for a, b_arm in _MCNEMAR_PAIRS:
        a_only, b_only, n_pairs = _paired_discordants(ok, a, b_arm)
        key = f"mcnemar_{a}_vs_{b_arm}"
        out[f"{key}_n"] = n_pairs
        out[f"{key}_first_only"] = a_only
        out[f"{key}_second_only"] = b_only
        out[f"{key}_p"] = _two_sided_binom_p(a_only, b_only)
    # Attributable fraction (e-s)/(e-f) on the common support of the three arms
    # (remediation plan 1b: the share of the episodic effect genuinely due to
    # conditioning content rather than the frequency confound).
    tri = [
        t for t in ok
        if all(
            t["arms"].get(lbl, {}).get("recovered") is not None
            for lbl in ("floor", "episodic_use", "swap")
        )
    ]
    out["attrib_n"] = len(tri)
    if tri:
        def tri_rate(lbl: str) -> float:
            return sum(bool(t["arms"][lbl]["recovered"]) for t in tri) / len(tri)

        e, s, f = tri_rate("episodic_use"), tri_rate("swap"), tri_rate("floor")
        out["attrib_rate_episodic"] = e
        out["attrib_rate_swap"] = s
        out["attrib_rate_floor"] = f
        if e != f:  # guard: fraction undefined when episodic == floor
            out["attributable_fraction"] = (e - s) / (e - f)
    out["swap_inapplicable"] = sum(
        1 for t in ok if "skipped" in t["arms"].get("swap", {})
    )
    out["a2_tail_inapplicable"] = sum(
        1 for t in ok if "skipped" in t["arms"].get("a2_tail", {})
    )
    return out


def _fmt_metrics(m: dict[str, float]) -> str:
    lines = ["", f"=== CLAMP RUN METRICS (N={int(m['n_ok'])}/{int(m['n_total'])}) ==="]
    lines.append(f"{'arm':<16}{'recovery':>14}  {'wilson95':>16}")
    for label in (*_PRIMARY_ARMS, *_C1_ARMS):
        r, d = int(m[f"recovery_{label}_n"]), int(m[f"denom_{label}"])
        lo, hi = m[f"recovery_{label}_wilson_lo"], m[f"recovery_{label}_wilson_hi"]
        lines.append(f"{label:<16}{r:>4}/{d:<4} = {m[f'recovery_{label}']:.3f}"
                     f"  [{lo:.3f}, {hi:.3f}]")
    lines.append("")
    lines.append(f"beyond-prompt (adapter recovers where floor AND clamped-prompt fail): "
                 f"episodic={int(m['beyond_prompt_episodic'])} dump={int(m['beyond_prompt_dump'])}")
    lines.append(f"McNemar floor vs episodic_use: adapter_only={int(m['mcnemar_adapter_only'])} "
                 f"floor_only={int(m['mcnemar_floor_only'])} p={m['mcnemar_p']:.4f}")
    for a, b_arm in _MCNEMAR_PAIRS:
        key = f"mcnemar_{a}_vs_{b_arm}"
        lines.append(f"McNemar {a} vs {b_arm}: n={int(m[f'{key}_n'])} "
                     f"{a}_only={int(m[f'{key}_first_only'])} "
                     f"{b_arm}_only={int(m[f'{key}_second_only'])} p={m[f'{key}_p']:.4f}")
    if "attributable_fraction" in m:
        lines.append(f"attributable fraction (e-s)/(e-f) on n={int(m['attrib_n'])}: "
                     f"{m['attributable_fraction']:.3f} "
                     f"(e={m['attrib_rate_episodic']:.3f} s={m['attrib_rate_swap']:.3f} "
                     f"f={m['attrib_rate_floor']:.3f})")
    lines.append(f"swap-inapplicable rows: {int(m['swap_inapplicable'])}")
    lines.append(f"a2_tail-inapplicable rows: {int(m['a2_tail_inapplicable'])}")
    return "\n".join(lines)


def _load_stratified(levels: list[str], per_level: int, offset: int = 0) -> list[Any]:
    from rune.bench.repobench import load_repobench_rows  # noqa: PLC0415

    rows: list[Any] = []
    for lvl in levels:
        rows.extend(load_repobench_rows(level=lvl)[offset : offset + per_level])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="8k,32k")
    ap.add_argument("--per-level", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N rows per level (use fresh rows uncontaminated by HPO tuning)")
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    # Episodic adapter config — defaults are the template-HPO winner.
    ap.add_argument("--variant", default=_EPISODIC_VARIANT)
    ap.add_argument("--anchor", type=int, default=_EPISODIC_ANCHOR)
    ap.add_argument("--scaling", type=float, default=_EPISODIC_SCALING)
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
    rows = _load_stratified(levels, args.per_level, args.offset)
    print(f"RepoBench rows: {len(rows)} (levels={levels} x {args.per_level}, offset={args.offset}, W={args.window})", flush=True)
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
        "offset": args.offset,
        "n_tasks": len(rows),
        "checkpoint_sha256": ckpt_sha,
        "engine_commit": engine_commit,
        "a2_full_max_tokens": _A2_FULL_MAX_TOKENS,
        "episodic_variant": args.variant,
        "episodic_anchor": args.anchor,
        "episodic_scaling": args.scaling,
    }
    run_name = (
        f"clamp-{args.variant}-W{args.window}-{'_'.join(levels)}"
        f"-n{len(rows)}-off{args.offset}-seed{args.seed}"
    )
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
