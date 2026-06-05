"""Generate LiveCodeBench solutions with the rune runner (REMOVE-BEFORE-MERGE).

Bridges LiveCodeBench v6 problems -> rune BenchTasks -> the rune engine, and dumps
{question_id, code_list:[code]} for grading by the OFFICIAL LCB harness
(tools/_lcb_grade.py, run in the isolated lcbenv). Minimal rune adaptation: the
engine just produces code; LCB does the input/output grading (functional + stdin).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

LCB_JSONL = "/tmp/lcb/test6.jsonl"
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
ARMS = {
    "scale0": {"checkpoint": C3_CKPT, "adapter_scaling": 0.0},
    "c3": {"checkpoint": C3_CKPT, "adapter_scaling": 1.0},
}


def _public_asserts(row: dict) -> str:
    """Public-example checks as asserts (functional only) for the engine's in-loop
    sandbox; LCB does the authoritative grading separately."""
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    fn = meta.get("func_name")
    if not fn:
        return ""  # stdin problem: no function to assert; engine runs code alone
    lines = ["from typing import List", "_s = Solution()"]
    for t in json.loads(row["public_test_cases"]):
        try:
            args = [ast.literal_eval(a) for a in t["input"].split("\n") if a.strip()]
            out = ast.literal_eval(t["output"])
        except Exception:
            continue
        call = f"_s.{fn}(*{args!r})"
        lines.append(f"assert {call} == {out!r}, {t['input']!r}")
    return "\n".join(lines)


def _to_task(row: dict) -> dict:
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    fn = meta.get("func_name", "")
    desc = row["question_content"]
    if row.get("starter_code"):
        desc += "\n\nComplete this starter code:\n" + row["starter_code"]
    return {
        "task_id": row["question_id"],
        "description": desc,
        "test_code": _public_asserts(row),
        "entry_point": fn,
        "signature": row.get("starter_code", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--functional-only", action="store_true")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--prompt-mode", default="full", dest="prompt_mode")
    ap.add_argument("--start-date", default="2025-02-01")
    ap.add_argument("--end-date", default="2025-05-01")
    args = ap.parse_args()

    import asyncio  # noqa: PLC0415

    from rune.bench.runner import BenchTask, run_benchmark  # noqa: PLC0415
    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    rows = [json.loads(line) for line in Path(LCB_JSONL).read_text().splitlines()]
    rows = [r for r in rows if args.start_date <= r["contest_date"][:10] < args.end_date]
    if args.functional_only:
        rows = [r for r in rows if r.get("starter_code")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"LCB problems: {len(rows)} (functional_only={args.functional_only})", flush=True)

    a = ARMS[args.arm]
    cfg = load_rune_config(None).override(
        checkpoint_path=a["checkpoint"],
        adapter_scaling=a["adapter_scaling"],
        seed=0,
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
            entry_point=t["entry_point"],
            signature=t["signature"],
        )
        for t in (_to_task(r) for r in rows)
    ]
    config: dict[str, Any] = {"model": model, "run_config": cfg.to_dict()}
    result = asyncio.run(run_benchmark(tasks, engine, config))
    gens = [
        {"question_id": r.task_id, "code_list": [r.code or ""]}
        for r in result.per_task
    ]
    Path(args.out).write_text(json.dumps(gens, indent=1))
    print(
        f"{args.arm}: wrote {len(gens)} generations -> {args.out} "
        f"(rune-internal pass@1={result.pass_at_1:.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
