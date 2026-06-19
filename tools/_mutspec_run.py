"""Mutated-spec pointer-vs-content control (issue #52 recall thesis).

c3 solves a subset of MBPP held-out tasks "spec-absent" (reference_a: the prompt
names only the function; the spec lives in the hypernetwork adapter conditioning).
This control mutates the spec so the correct answer changes, re-runs c3 spec-absent,
and classifies each output:
  CONTENT  = passes the MUTATED tests  (tracked the mutated spec -> genuine recall)
  POINTER  = fails mutated, passes ORIGINAL tests (reproduced the memorized solution)
  OTHER    = fails both
Run c3 with the adapter ON (reference_a, scaling 0.627 — the HPO best that produced
c3's 14/24 spec-absent; the 8/24 figure is the adapter-OFF floor, not c3).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
HELDOUT = "benchmarks/mbpp_heldout_tasks.json"


def _expected_values(test_code: str) -> list[Any]:
    """Literal RHS of each top-level `assert call == <expected>` line."""
    vals: list[Any] = []
    for line in test_code.splitlines():
        s = line.strip()
        if not s.startswith("assert ") or "==" not in s:
            continue
        try:
            node = ast.parse(s).body[0]
            cmp = node.test  # type: ignore[attr-defined]
            vals.append(ast.literal_eval(cmp.comparators[0]))
        except (SyntaxError, ValueError, AttributeError, IndexError):
            return []
    return vals


def _mutation(expected: list[Any]) -> tuple[str, Any] | None:
    """Pick a deterministic, type-aware mutation that changes EVERY expected value.

    Returns (instruction, transform_fn) or None if no clean mutation applies.
    """
    if not expected:
        return None

    def all_is(typ: type) -> bool:
        return all(isinstance(e, typ) and not isinstance(e, bool) for e in expected)

    # bool first (bool is subclass of int)
    if all(isinstance(e, bool) for e in expected):
        return ("Return the BOOLEAN NEGATION of the result described above.", lambda x: not x)
    if all_is(int) or all_is(float):
        return ("Add 1 to the result described above before returning it.", lambda x: x + 1)
    if all(isinstance(e, str) for e in expected):
        if any(len(e) < 2 or e == e[::-1] for e in expected):
            return None
        return ("Return the result described above REVERSED (the string backwards).", lambda x: x[::-1])
    if all(isinstance(e, (list, tuple)) for e in expected):
        if any(len(e) < 2 or list(e) == list(e)[::-1] for e in expected):
            return None
        return (
            "Return the sequence result described above in REVERSE order.",
            lambda x: type(x)(list(x)[::-1]),
        )
    return None


def mutate_task(task: dict[str, Any]) -> dict[str, Any] | None:
    """Build a mutated task (description + test_code) or None if not cleanly mutable."""
    expected = _expected_values(task["test_code"])
    mut = _mutation(expected)
    if mut is None:
        return None
    instr, fn = mut
    # mutate every assert's RHS
    new_lines = []
    for line in task["test_code"].splitlines():
        s = line.strip()
        if s.startswith("assert ") and "==" in s:
            try:
                node = ast.parse(s).body[0]
                cmp = node.test  # type: ignore[attr-defined]
                old = ast.literal_eval(cmp.comparators[0])
                new = fn(old)
                left = s[len("assert "):].split("==")[0].strip()
                new_lines.append(f"assert {left} == {new!r}")
                continue
            except Exception:
                return None
        new_lines.append(line)
    mutated_test = "\n".join(new_lines)
    # mutate the description (prose + drop the now-wrong doctest, add the instruction)
    desc = task["description"]
    desc_no_doctest = "\n".join(ln for ln in desc.splitlines() if ">>>" not in ln)
    mutated_desc = desc_no_doctest.rstrip().rstrip('"').rstrip() + f"\n\n{instr}\n\"\"\"\n"
    return {**task, "description": mutated_desc, "test_code": mutated_test, "_mut_instr": instr}


async def _run_ref_a(tasks: list[dict[str, Any]], cfg: Any, model: Any) -> list[Any]:
    from rune.bench.runner import BenchTask, run_benchmark  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415

    bts = [
        BenchTask(
            task_id=t["task_id"], description=t["description"], test_code=t["test_code"],
            entry_point=t["entry_point"], signature=t.get("signature", ""),
            public_checks=t.get("public_checks", ""),
        )
        for t in tasks
    ]
    engine = create_engine()
    config = {"model": model, "run_config": cfg.to_dict(), "benchmark": "mbpp_recall_mutspec"}
    res = await run_benchmark(bts, engine, config)
    return list(res.per_task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", type=float, default=0.627)
    ap.add_argument("--out", default="/tmp/lcbout/mutspec_summary.json")
    ap.add_argument("--experiment", default="issue52-mutspec-control")
    args = ap.parse_args()

    import mlflow  # noqa: PLC0415

    from rune.config import load_rune_config  # noqa: PLC0415
    from rune.engine.continuation import strip_self_tests  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415
    from rune.tracking import configure_mlflow, tracked_run  # noqa: PLC0415

    cfg = load_rune_config(None).override(
        checkpoint_path=C3_CKPT, adapter_scaling=args.scaling, seed=0,
        prompt_mode="reference_a",
    )
    model = ModelWrapper.from_config(cfg)
    tasks = json.loads(Path(HELDOUT).read_text())

    # 1) c3 spec-absent on ORIGINAL specs -> identify solves
    orig = asyncio.run(_run_ref_a(tasks, cfg, model))
    by_id = {t["task_id"]: t for t in tasks}
    solves = [r.task_id for r in orig if r.passed]
    print(f"c3 spec-absent solves (reference_a@{args.scaling}): {len(solves)}/{len(tasks)} -> {solves}", flush=True)

    # 2) mutate the solved tasks
    mutated = []
    skipped = []
    for tid in solves:
        m = mutate_task(by_id[tid])
        (mutated.append(m) if m else skipped.append(tid))
    print(f"mutable solves: {len(mutated)}; skipped (no clean mutation): {skipped}", flush=True)

    # 3) c3 spec-absent on MUTATED specs
    mut_res = asyncio.run(_run_ref_a(mutated, cfg, model)) if mutated else []
    mut_by_id = {m["task_id"]: m for m in mutated}

    # 4) classify
    rows = []
    for r in mut_res:
        t = mut_by_id[r.task_id]
        code = strip_self_tests(r.code or "")
        passed_mut = bool(r.passed)
        passed_orig = run_in_sandbox(code + "\n\n" + by_id[r.task_id]["test_code"], timeout=30).exit_code == 0
        label = "content" if passed_mut else ("pointer" if passed_orig else "other")
        rows.append({"task_id": r.task_id, "label": label, "passed_mutated": passed_mut,
                     "passed_original": passed_orig, "mutation": t["_mut_instr"]})
        print(f"  {r.task_id}: {label} (mut={passed_mut} orig={passed_orig})", flush=True)

    n = len(rows)
    content = sum(1 for x in rows if x["label"] == "content")
    pointer = sum(1 for x in rows if x["label"] == "pointer")
    other = sum(1 for x in rows if x["label"] == "other")
    summary = {
        "scaling": args.scaling, "n_solves": len(solves), "solves": solves,
        "n_mutated": n, "skipped": skipped,
        "content": content, "pointer": pointer, "other": other,
        "content_frac": (content / n if n else None),
        "pointer_frac": (pointer / n if n else None),
        "rows": rows,
        "interpretation": "content = tracks mutated spec (genuine recall); pointer = reproduces memorized original (confound).",
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nMUTSPEC: content={content} pointer={pointer} other={other} (n={n})", flush=True)

    ckpt_sha = hashlib.sha256(Path(C3_CKPT).read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False, cwd=str(Path(__file__).resolve().parent.parent)).stdout.strip()
    configure_mlflow(args.experiment)
    with tracked_run(f"mutspec-c3-ref_a-s{args.scaling}", params={
        **cfg.to_dict(), "benchmark": "mbpp_recall_mutspec", "checkpoint_sha256": ckpt_sha,
        "engine_commit": commit, "n_solves": len(solves), "n_mutated": n,
    }):
        mlflow.log_metric("content", content)
        mlflow.log_metric("pointer", pointer)
        mlflow.log_metric("other", other)
        if n:
            mlflow.log_metric("content_frac", content / n)
        mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()
