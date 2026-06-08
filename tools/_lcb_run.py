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
    "scale0": {"checkpoint": C3_CKPT, "adapter_scaling": 0.0},
    "c3": {"checkpoint": C3_CKPT, "adapter_scaling": 1.0},
}


def _to_task(row: dict[str, Any]) -> dict[str, Any]:
    from rune.bench.lcb import build_public_assert_checks  # noqa: PLC0415

    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    fn = meta.get("func_name", "")
    desc = row["question_content"]
    if row.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + row["starter_code"]
    public = build_public_assert_checks(row)
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
    if args.limit:
        rows = rows[: args.limit]
    return rows


def _official_lcb_pass_at_1(gens_path: Path, *, timeout: int = 6) -> float | None:
    """Run the official LCB grader in lcbenv; return pass@1 or None if unavailable."""
    grade_script = Path(__file__).resolve().parent / "_lcb_grade.py"
    if not LCB_GRADE_PYTHON.is_file() or not grade_script.is_file():
        return None
    env = {**os.environ, "PYTHONPATH": str(LCB_HARNESS)}
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
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr, flush=True)
        return None
    match = re.search(r"LCB pass@1 = (\S+)", proc.stdout)
    if not match:
        return None
    return float(match.group(1))


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
    ap.add_argument("--functional-only", action="store_true")
    ap.add_argument("--max-iters", type=int, default=4)
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
    scaling = a["adapter_scaling"] if args.adapter_scaling is None else args.adapter_scaling
    cfg = load_rune_config(None).override(
        checkpoint_path=a["checkpoint"],
        adapter_scaling=scaling,
        seed=args.seed,
        max_phase_iterations=args.max_iters,
        prompt_mode=args.prompt_mode,
        model_judge=False,
    )
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
        for t in (_to_task(r) for r in rows)
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
    configure_mlflow(args.experiment)
    params = {
        **cfg.to_dict(),
        "arm": args.arm,
        "benchmark": "livecodebench_v6",
        "lcb_jsonl": LCB_JSONL,
        "lcb_sha256": lcb_sha,
        "functional_only": args.functional_only,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_tasks": len(tasks),
    }
    run_name = f"lcb-{args.arm}-{args.prompt_mode}-seed{args.seed}"
    with tracked_run(run_name, params=params):
        log_dataset(Path(LCB_JSONL), name="test6.jsonl", context="test")
        result = asyncio.run(
            run_benchmark(tasks, engine, config, sessions_dir=sessions)
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

        if args.grade:
            official = _official_lcb_pass_at_1(out_path)
            if official is not None:
                mlflow.log_metric("lcb_official_pass_at_1", official)
                print(f"LCB official pass@1={official:.3f} (n={len(gens)})", flush=True)
            else:
                print(
                    "Official LCB grading skipped (lcbenv unavailable or failed)",
                    flush=True,
                )

    print(
        f"{args.arm}: wrote {len(gens)} generations -> {out_path} "
        f"(rune-internal pass@1={result.pass_at_1:.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
