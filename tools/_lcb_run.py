"""Generate LiveCodeBench solutions with the rune runner (REMOVE-BEFORE-MERGE).

Bridges LiveCodeBench v6 problems -> rune BenchTasks -> the rune engine, and dumps
{question_id, code_list:[code]} for grading by the OFFICIAL LCB harness
(tools/_lcb_grade.py, run in the isolated lcbenv). Minimal rune adaptation: the
engine just produces code; LCB does the input/output grading (functional + stdin).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

LCB_JSONL = "/tmp/lcb/test6.jsonl"
LCB_GRADE_PYTHON = Path("/tmp/lcbenv/bin/python")
LCB_HARNESS = Path("/tmp/LiveCodeBench")
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
ARMS = {
    # base: single-shot capability ceiling — adapter OFF, NO engine loop. The c3
    # checkpoint is loaded but never hot-swapped (scaling 0 == base weights). This
    # is the engine's escalate zero-shot first attempt in isolation (same
    # prompt_zeroshot template + "code" action system prompt, freeform), so c3/
    # scale0 are strict supersets of base by construction.
    "base": {"checkpoint": C3_CKPT, "adapter_scaling": 0.0},
    "scale0": {"checkpoint": C3_CKPT, "adapter_scaling": 0.0},
    "c3": {"checkpoint": C3_CKPT, "adapter_scaling": 1.0},
}


def _to_task(
    row: dict[str, Any], *, merge_spec_public_checks: bool = False
) -> dict[str, Any]:
    from rune.bench.lcb import build_public_assert_checks  # noqa: PLC0415

    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    fn = meta.get("func_name", "")
    desc = row["question_content"]
    if row.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + row["starter_code"]
    public = build_public_assert_checks(
        row, merge_spec_public_checks=merge_spec_public_checks
    )
    return {
        "task_id": row["question_id"],
        "description": desc,
        "test_code": public,
        "public_checks": public,
        "entry_point": fn,
        "signature": row.get("starter_code", ""),
    }


def _load_lcb_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(LCB_JSONL).read_text().splitlines()]
    rows = [
        r for r in rows if args.start_date <= r["contest_date"][:10] < args.end_date
    ]
    if args.functional_only:
        rows = [r for r in rows if r.get("starter_code")]
    if args.qids:
        want = {q.strip() for q in args.qids.split(",") if q.strip()}
        rows = [r for r in rows if r["question_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def _official_lcb_pass_at_1(
    gens_path: Path, *, timeout: int = 6
) -> tuple[float | None, str]:
    """Run the official LCB grader in lcbenv.

    Returns ``(pass_at_1, raw_output)`` — pass_at_1 is None if the grader is
    unavailable or failed; raw_output is the grader's stdout/stderr (logged as a
    durable MLflow artifact so the official grade is reproducible, not a comment).
    """
    grade_script = Path(__file__).resolve().parent / "_lcb_grade.py"
    if not LCB_GRADE_PYTHON.is_file() or not grade_script.is_file():
        return None, "grader unavailable (lcbenv/_lcb_grade.py missing)"
    repo_src = Path(__file__).resolve().parent.parent / "src"
    env = {
        **os.environ,
        "PYTHONPATH": f"{repo_src}:{LCB_HARNESS}",
    }
    proc = subprocess.run(
        [
            str(LCB_GRADE_PYTHON),
            str(grade_script),
            "--gens",
            str(gens_path),
            "--timeout",
            str(timeout),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    out = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr, flush=True)
        return None, out
    match = re.search(r"LCB pass@1 = (\S+)", proc.stdout)
    if not match:
        return None, out
    return float(match.group(1)), out


async def _generate_base(model: Any, tasks: list[Any], cfg: Any) -> list[dict[str, Any]]:
    """Single-shot base capability ceiling: one generate per task, adapter off.

    Mirrors the engine's escalate zero-shot attempt exactly — the same
    ``prompt_zeroshot`` template and the "code" action's freeform system prompt —
    but with NO decompose/plan/diagnose/repair loop. This is the headline base
    arm; c3/scale0 (which start from this same zero-shot) are strict supersets.
    """
    import torch  # noqa: PLC0415

    from rune.bench.lcb import normalize_lcb_submission  # noqa: PLC0415
    from rune.engine.continuation import extract_partial_code  # noqa: PLC0415
    from rune.engine.parse import render_template  # noqa: PLC0415

    rc = cfg.to_dict()
    seed = rc.get("seed")
    gens: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        # Match run_benchmark's per-task seeding (seed + i) so base is
        # deterministic/reproducible and uses the same RNG convention as the
        # c3/scale0 engine arms (temperature 0.3 is stochastic otherwise).
        if seed is not None:
            torch.manual_seed(seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + i)
        prompt = render_template(
            "prompt_zeroshot",
            task_description=t.description,
            entry_point=t.entry_point,
        )
        gen = await model.generate(
            prompt=prompt,
            system_prompt="You are a code generator.",
            output_schema=None,
            max_tokens=rc.get("max_tokens", 2048),
            temperature=rc.get("temperature", 0.3),
            repetition_penalty=rc.get("repetition_penalty", 1.1),
            top_p=rc.get("top_p", 0.9),
            no_repeat_ngram_size=rc.get("no_repeat_ngram_size", 0),
            presence_penalty=rc.get("presence_penalty", 0.0),
            thinking_budget=rc.get("thinking_budget", 0),
        )
        code = extract_partial_code(gen.text)
        gens.append(
            {
                "question_id": t.task_id,
                "code_list": [
                    normalize_lcb_submission(
                        code, t.entry_point, _starter_code=t.signature
                    )
                ],
            }
        )
        print(f"base {t.task_id}: gen {len(code)} chars", flush=True)
    return gens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--sessions",
        default=None,
        help="Per-task session.jsonl dir (default: <out_stem>_sessions beside --out)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--qids",
        default="",
        help="Comma-separated question_id filter (e.g. 3753,3777)",
    )
    ap.add_argument("--functional-only", action="store_true")
    ap.add_argument(
        "--max-iters",
        type=int,
        default=24,
        help="Engine max_phase_iterations budget; 24 is the proper-budget "
        "value that lets escalate fully exhaust the repair loop.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-mode", default="full", dest="prompt_mode")
    ap.add_argument(
        "--adapter-scaling",
        type=float,
        default=None,
        help="Override the arm's adapter_scaling (sweep/HPO best is 0.627).",
    )
    ap.add_argument("--start-date", default="2025-02-01")
    ap.add_argument("--end-date", default="2025-05-01")
    ap.add_argument("--experiment", default="issue52-goal3-lcb")
    ap.add_argument(
        "--grade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After generation, run official LCB grading and log pass@1 to MLflow",
    )
    ap.add_argument(
        "--repair-brief-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--plan-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--replan-on-complexity",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument("--max-repairs", type=int, default=None)
    # Budget guards (issue #52 §4 levers 3+4); all default OFF for pre-registration
    # safety. Passing all three enables the full guard set for a run.
    ap.add_argument(
        "--repair-dedup-after",
        type=int,
        default=None,
        help="Stop retrying a subtask after N consecutive identical "
        "(stderr, approach) failing attempts (default off).",
    )
    ap.add_argument(
        "--complexity-repair-cap",
        type=int,
        default=None,
        help="Stop retrying a subtask after K consecutive complexity-only "
        "rejections (default off).",
    )
    ap.add_argument(
        "--continuation-structural-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Abort continuation on a prose chunk once a salvageable entry "
        "function exists (default off).",
    )
    ap.add_argument(
        "--repair-context-fix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render repair brief + last failure in the thin repair prompt and "
        "tail-cut history errors so the assert payload survives (default off).",
    )
    ap.add_argument(
        "--concise-code-instruction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Instruct code/repair prompts to output code directly with minimal "
        "comments — keeps chain-of-thought out of the completion (default off).",
    )
    ap.add_argument(
        "--adapter-cond-budget-fix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pack adapter conditioning into the hypernet's 2048-token encoder "
        "window so Review Feedback is never silently truncated (default off).",
    )
    ap.add_argument(
        "--model-judge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the correctness judge on oracle-passing units (oracle has "
        "precedence: judge only flips pass->fail with a grounded failing input).",
    )
    args = ap.parse_args()

    import asyncio  # noqa: PLC0415

    import mlflow  # noqa: PLC0415

    from rune.bench.lcb import normalize_lcb_submission  # noqa: PLC0415
    from rune.bench.runner import BenchTask, run_benchmark  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import (  # noqa: PLC0415
        configure_mlflow,
        log_dataset,
        tracked_run,
    )

    rows = _load_lcb_rows(args)
    print(
        f"LCB problems: {len(rows)} (functional_only={args.functional_only})",
        flush=True,
    )

    a = ARMS[args.arm]
    scaling = (
        a["adapter_scaling"] if args.adapter_scaling is None else args.adapter_scaling
    )
    overrides: dict[str, Any] = {
        "checkpoint_path": a["checkpoint"],
        "adapter_scaling": scaling,
        "seed": args.seed,
        "max_phase_iterations": args.max_iters,
        "prompt_mode": args.prompt_mode,
    }
    if args.repair_brief_enabled is not None:
        overrides["repair_brief_enabled"] = args.repair_brief_enabled
    if args.plan_gate_enabled is not None:
        overrides["plan_gate_enabled"] = args.plan_gate_enabled
    if args.replan_on_complexity is not None:
        overrides["replan_on_complexity"] = args.replan_on_complexity
    if args.max_repairs is not None:
        overrides["max_repairs"] = args.max_repairs
    if args.repair_dedup_after is not None:
        overrides["repair_dedup_after"] = args.repair_dedup_after
    if args.complexity_repair_cap is not None:
        overrides["complexity_repair_cap"] = args.complexity_repair_cap
    if args.continuation_structural_guard is not None:
        overrides["continuation_structural_guard"] = args.continuation_structural_guard
    if args.repair_context_fix is not None:
        overrides["repair_context_fix"] = args.repair_context_fix
    if args.concise_code_instruction is not None:
        overrides["concise_code_instruction"] = args.concise_code_instruction
    if args.adapter_cond_budget_fix is not None:
        overrides["adapter_cond_budget_fix"] = args.adapter_cond_budget_fix
    if args.model_judge is not None:
        overrides["model_judge"] = args.model_judge
    cfg = load_rune_config(None).override(**overrides)
    model = ModelWrapper.from_config(cfg)
    engine = create_engine()
    tasks = [
        BenchTask(
            task_id=t["task_id"],
            description=t["description"],
            test_code=t["test_code"],
            public_checks=t["public_checks"],
            entry_point=t["entry_point"],
            signature=t["signature"],
        )
        for t in (
            _to_task(r, merge_spec_public_checks=cfg.merge_spec_public_checks)
            for r in rows
        )
    ]
    config: dict[str, Any] = {
        "model": model,
        "run_config": cfg.to_dict(),
        "benchmark": "livecodebench_v6",
    }
    out_path = Path(args.out)
    sessions = Path(args.sessions or out_path.parent / f"{out_path.stem}_sessions")
    sessions.mkdir(parents=True, exist_ok=True)

    lcb_sha = hashlib.sha256(Path(LCB_JSONL).read_bytes()).hexdigest()
    ckpt_sha = hashlib.sha256(Path(a["checkpoint"]).read_bytes()).hexdigest()
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
        "arm": args.arm,
        "benchmark": "livecodebench_v6",
        "lcb_jsonl": LCB_JSONL,
        "lcb_sha256": lcb_sha,
        "checkpoint_sha256": ckpt_sha,
        "engine_commit": engine_commit,
        "timeout_s": 6,
        "functional_only": args.functional_only,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_tasks": len(tasks),
    }
    run_name = f"lcb-{args.arm}-{args.prompt_mode}-seed{args.seed}"
    result = None
    with tracked_run(run_name, params=params):
        log_dataset(Path(LCB_JSONL), name="test6.jsonl", context="test")
        if args.arm == "base":
            gens = asyncio.run(_generate_base(model, tasks, cfg))
            mlflow.log_metric("total_tasks", len(gens))
        else:
            result = asyncio.run(
                run_benchmark(tasks, engine, config, sessions_dir=sessions, resume=True)
            )
            mlflow.log_metric("pass_at_1", result.pass_at_1)
            mlflow.log_metric("passed_tasks", result.passed_tasks)
            mlflow.log_metric("total_tasks", result.total_tasks)
            gens = [
                {
                    "question_id": r.task_id,
                    "code_list": [
                        normalize_lcb_submission(
                            r.code or "",
                            t.entry_point,
                            _starter_code=t.signature,
                        )
                    ],
                }
                for r, t in zip(result.per_task, tasks, strict=True)
            ]
        out_path.write_text(json.dumps(gens, indent=1))
        mlflow.log_artifact(str(out_path))  # durable: the graded generations

        if args.grade:
            official, grade_out = _official_lcb_pass_at_1(out_path)
            grade_path = out_path.with_suffix(".grade.txt")
            grade_path.write_text(grade_out)
            mlflow.log_artifact(str(grade_path))  # durable: official grade output
            if official is not None:
                mlflow.log_metric("lcb_official_pass_at_1", official)
                print(f"LCB official pass@1={official:.3f} (n={len(gens)})", flush=True)
            else:
                print(
                    "Official LCB grading skipped (lcbenv unavailable or failed)",
                    flush=True,
                )

    internal = (
        f" (rune-internal pass@1={result.pass_at_1:.3f})" if result is not None else ""
    )
    print(f"{args.arm}: wrote {len(gens)} generations -> {out_path}{internal}", flush=True)


if __name__ == "__main__":
    main()
