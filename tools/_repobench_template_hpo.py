"""Optuna HPO over episodic adapter templates (issue #52 long-context).

Squeezes the corrected episodic per-task adapter template: tunes the template
variant x in-file anchor x adapter scaling to maximize cross-file API recovery on
a TUNING set, then reports the best config on a held-out set it never saw (plus
floor / a2_full / dump_gf baselines). Pool is mixed 8k+32k under a clamped window
(the constrained-hardware regime) — episodic conditioning stays small even at 32k.

Objective (maximize): soft-recovery = mean over tasks of (1.0 if the gold
cross-file identifier was recovered else edit-similarity) — recovery is sparse at
small N, so es gives Optuna a smooth gradient.

Caching: the adapter BUILD (hypernet forward) is the expensive step and depends
only on (task, variant, anchor) — cached and reused across trials; scaling is a
cheap re-hotswap.

Run: uv run --extra gpu python tools/_repobench_template_hpo.py \
       --n-8k 15 --n-32k 15 --trials 24 --window 768 --experiment issue52-repobench-template-hpo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
_COND_CHAR_CAP = 16000
_A2_FULL_MAX_TOKENS = 12000

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


def _soft(pred: str, row: Any) -> tuple[float, bool, float]:
    from rune.bench.identifier_match import (  # noqa: PLC0415
        edit_similarity,
        gold_id_recovery,
    )

    es = edit_similarity(pred, row.next_line)
    rec = (
        bool(gold_id_recovery(pred, row.gold_identifier))
        if row.gold_identifier
        else False
    )
    return (1.0 if rec else es), rec, round(es, 3)


def main() -> None:  # noqa: C901, PLR0915 - linear experiment script
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-8k", type=int, default=15)
    ap.add_argument("--n-32k", type=int, default=15)
    ap.add_argument(
        "--offset-8k", type=int, default=40, help="skip the headline/smoke 8k rows"
    )
    ap.add_argument("--offset-32k", type=int, default=10)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--window", type=int, default=768)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tuning-fraction", type=float, default=0.667)
    ap.add_argument("--experiment", default="issue52-repobench-template-hpo")
    ap.add_argument("--out", default="/tmp/rb_template_hpo.json")
    args = ap.parse_args()

    import os  # noqa: PLC0415

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import asyncio  # noqa: PLC0415

    import mlflow  # noqa: PLC0415
    import optuna  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from rune.bench.hpo import split_tasks  # noqa: PLC0415
    from rune.bench.repobench import (  # noqa: PLC0415
        EPISODIC_VARIANTS,
        load_repobench_rows,
        render_context_prompt,
        render_episodic,
        render_xfile_adapter,
    )
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.model.adapter import scale_lora_b  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    rows8 = load_repobench_rows(level="8k")[args.offset_8k : args.offset_8k + args.n_8k]
    rows32 = load_repobench_rows(level="32k")[
        args.offset_32k : args.offset_32k + args.n_32k
    ]
    pool = rows8 + rows32
    tune, hold = split_tasks(pool, seed=args.seed, tuning_fraction=args.tuning_fraction)
    print(
        f"pool={len(pool)} (8k={len(rows8)} 32k={len(rows32)}) -> tune={len(tune)} holdout={len(hold)}",
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

    floor_p = {
        r.task_id: model.clamp_to_window(
            f"# Current file:\n{_prefix(r)}\n# Next line:", args.window
        )
        for r in pool
    }
    build_cache: dict[tuple[str, str, int], Any] = {}

    def adapter_sd(row: Any, variant: str, anchor: int) -> Any:
        key = (row.task_id, variant, anchor)
        if key not in build_cache:
            cond = render_episodic(row, variant, anchor_chars=anchor)[:_COND_CHAR_CAP]
            build_cache[key] = model.generate_adapter(cond).state_dict
        return build_cache[key]

    async def eval_adapter(
        rows: list[Any], variant: str, anchor: int, scaling: float
    ) -> tuple[float, list[dict[str, Any]]]:
        total, recs = 0.0, []
        for row in rows:
            model.hotswap_adapter(
                scale_lora_b(adapter_sd(row, variant, anchor), scaling)
            )
            torch.manual_seed(args.seed)
            pred = await _gen_line(model, floor_p[row.task_id], args.max_new)
            soft, rec, es = _soft(pred, row)
            total += soft
            recs.append(
                {
                    "task_id": row.task_id,
                    "level": row.level,
                    "gold": row.gold_identifier,
                    "pred": pred,
                    "recovered": rec,
                    "es": es,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return total / len(rows), recs

    def objective(trial: optuna.Trial) -> float:
        variant = trial.suggest_categorical("variant", list(EPISODIC_VARIANTS))
        anchor = trial.suggest_categorical("anchor_chars", [0, 400])
        scaling = trial.suggest_float("scaling", 0.4, 1.3)
        score, _ = asyncio.run(eval_adapter(tune, variant, anchor, scaling))
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_params
    print(
        f"\nBEST PARAMS: {best}  (tuning soft-recovery={study.best_value:.3f})",
        flush=True,
    )

    # ---- Held-out evaluation: best episodic config vs baselines ----
    async def eval_baselines(rows: list[Any]) -> dict[str, Any]:
        out: dict[str, list[dict[str, Any]]] = {
            "floor": [],
            "a2_full": [],
            "dump_gf": [],
        }
        for row in rows:
            ctx = render_context_prompt(row)
            a2p = f"# Cross-file context:\n{ctx}\n\n# Current file:\n{_prefix(row)}\n# Next line:"
            model.reset_adapter()
            torch.manual_seed(args.seed)
            fp = await _gen_line(model, floor_p[row.task_id], args.max_new)
            _, frec, fes = _soft(fp, row)
            out["floor"].append({"task_id": row.task_id, "recovered": frec, "es": fes})
            if model.count_tokens(a2p) <= _A2_FULL_MAX_TOKENS:
                model.reset_adapter()
                torch.manual_seed(args.seed)
                ap_pred = await _gen_line(model, a2p, args.max_new)
                _, arec, aes = _soft(ap_pred, row)
                out["a2_full"].append(
                    {"task_id": row.task_id, "recovered": arec, "es": aes}
                )
            dump = render_xfile_adapter(row, "structured", gold_first=True)[
                :_COND_CHAR_CAP
            ]
            model.hotswap_adapter(
                scale_lora_b(model.generate_adapter(dump).state_dict, 1.0)
            )
            torch.manual_seed(args.seed)
            dp = await _gen_line(model, floor_p[row.task_id], args.max_new)
            _, drec, des = _soft(dp, row)
            out["dump_gf"].append(
                {"task_id": row.task_id, "recovered": drec, "es": des}
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out

    best_score, best_recs = asyncio.run(
        eval_adapter(hold, best["variant"], best["anchor_chars"], best["scaling"])
    )
    baselines = asyncio.run(eval_baselines(hold))

    def rate(recs: list[dict[str, Any]]) -> tuple[int, int, float]:
        n = len(recs) or 1
        r = sum(1 for x in recs if x["recovered"])
        es = sum(x["es"] for x in recs) / n
        return r, len(recs), es

    print(f"\n=== HELD-OUT (N={len(hold)}, never tuned) ===", flush=True)
    rows_report = [("best_episodic", best_recs)] + [
        (k, v) for k, v in baselines.items()
    ]
    for name, recs in rows_report:
        r, d, es = rate(recs)
        print(f"  {name:<14} recovery {r}/{d}   mean_es={es:.3f}", flush=True)

    # ---- Durable MLflow ----
    ckpt_sha = hashlib.sha256(Path(C3_CKPT).read_bytes()).hexdigest()
    engine_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    ).stdout.strip()
    configure_mlflow(args.experiment)
    params = {
        **cfg.to_dict(),
        "benchmark": "repobench_v1.1_python",
        "split": "cross_file_first",
        "window": args.window,
        "n_pool": len(pool),
        "n_tune": len(tune),
        "n_holdout": len(hold),
        "trials": args.trials,
        "checkpoint_sha256": ckpt_sha,
        "engine_commit": engine_commit,
        "best_variant": best["variant"],
        "best_anchor_chars": best["anchor_chars"],
        "best_scaling": round(best["scaling"], 4),
    }
    with tracked_run(
        f"template-hpo-W{args.window}-n{len(pool)}-t{args.trials}", params=params
    ):
        mlflow.log_metric("tuning_best_soft_recovery", study.best_value)
        for name, recs in rows_report:
            r, d, es = rate(recs)
            safe = name.replace("@", "_at_")
            mlflow.log_metric(f"holdout_recovery_{safe}", r / (d or 1))
            mlflow.log_metric(f"holdout_es_{safe}", es)
        payload = {
            "best_params": best,
            "tuning_best": study.best_value,
            "trials": [{"params": t.params, "value": t.value} for t in study.trials],
            "holdout": {name: recs for name, recs in rows_report},
        }
        Path(args.out).write_text(json.dumps(payload, indent=1))
        mlflow.log_artifact(str(args.out))
    print(f"\nwrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
