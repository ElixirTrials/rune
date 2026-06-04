"""GOAL-3 step-1: multi-turn adapter-as-memory-substrate probe (REMOVE-BEFORE-MERGE).

Runs ON THE RUNE RUNNER (standing rule: never a parallel runner). The three arms
are three PipelineConfigs driven through `rune.bench.runner.run_benchmark` ->
`engine.ainvoke` (the real code->diagnose->repair* loop). Nothing in the generation
path is reimplemented here; only the parity check touches logits directly.

Pre-registration: docs/issue52-goal3-multiturn-substrate-2026-06-04.md (a-g + advisor
blockers B1-B3). Headline = success-vs-turn curve + recovery gap (final - attempt1)
on the scale=0-attempt-1-fail slice, scored POST-HOC vs held-out test_code.

Subcommands:
  parity  - forward-parity check (b): max|scale0-adapter - base| (expect 0).
  run     - drive one arm over a task pool; writes per-task session.jsonl + results.
  score   - post-hoc per-turn pass/fail (vs held-out tests) + token accounting (g).
  analyze - combine 3 arms, apply slice cut, bootstrap CIs on curve + recovery gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RUNE = Path(__file__).resolve().parents[1]
C3_CKPT = "/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt"
WARM_CKPT = str(
    RUNE
    / "third_party/doc-to-lora/trained_d2l/qwen_4b_d2l"
    / "checkpoint-20000/pytorch_model.bin"
)

# arm -> (checkpoint, adapter_scaling). scale0 zeroes lora_B on the SAME path
# (any ckpt works; delta is exactly 0 — verified by `parity`).
ARMS: dict[str, dict[str, Any]] = {
    "scale0": {"checkpoint": C3_CKPT, "adapter_scaling": 0.0},
    "warm": {"checkpoint": WARM_CKPT, "adapter_scaling": 1.0},
    "c3": {"checkpoint": C3_CKPT, "adapter_scaling": 1.0},
}

_CODE_ACTIONS = {"code", "repair", "integrate"}


def _build_cfg(arm: str, seed: int, max_iters: int, prompt_mode: str = "full") -> Any:
    from rune.config import load_rune_config  # noqa: PLC0415

    a = ARMS[arm]
    cfg = load_rune_config(None)
    return cfg.override(
        checkpoint_path=a["checkpoint"],
        adapter_scaling=a["adapter_scaling"],
        seed=seed,
        max_phase_iterations=max_iters,
        prompt_mode=prompt_mode,
        # Corrected runner uses the deterministic public-example oracle; the
        # in-loop judge false-positives correct code + is slow, so it is OFF for
        # these experiments (a separate, to-be-validated arm).
        model_judge=False,
    )


# --------------------------------------------------------------------------- parity
def cmd_parity(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415

    from rune.model.adapter import scale_lora_b  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    cfg = _build_cfg("c3", 0, 10)
    mw = ModelWrapper.from_config(cfg)
    bm = mw._base_model
    tok = mw._tokenizer
    device = next(bm.parameters()).device
    text = (
        "## Task\nWrite a python function to add two integers.\n\n"
        "## Current Code\ndef add(a, b):\n    return a - b\n\n"
        "## Review Feedback\nWrong operator; should add."
    )
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    adapter = mw.generate_adapter(text)

    def logits(scale: float) -> torch.Tensor:
        mw.hotswap_adapter(scale_lora_b(adapter.state_dict, scale))
        with torch.no_grad():
            return bm(ids).logits.float().cpu()

    l1 = logits(1.0)
    l0 = logits(0.0)
    with torch.no_grad(), bm.disable_adapter():
        lb = bm(ids).logits.float().cpu()

    d0 = (l0 - lb).abs().max().item()
    d1 = (l1 - lb).abs().max().item()
    out = {
        "max_abs_scale0_minus_base": d0,
        "max_abs_scale1_minus_base": d1,
        "parity_ok": d0 == 0.0,
        "adapter_nontrivial": d1 > 0.0,
    }
    print(json.dumps(out, indent=2))


# ------------------------------------------------------------------------------ run
def cmd_run(args: argparse.Namespace) -> None:
    import asyncio  # noqa: PLC0415
    import hashlib  # noqa: PLC0415

    import mlflow  # noqa: PLC0415

    from rune.bench.runner import load_tasks, run_benchmark  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415
    from rune.tracking import (  # noqa: PLC0415
        configure_mlflow,
        log_dataset,
        tracked_run,
    )

    cfg = _build_cfg(args.arm, args.seed, args.max_iters, args.prompt_mode)
    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]
    model = ModelWrapper.from_config(cfg)
    engine = create_engine()
    config = {"model": model, "run_config": cfg.to_dict()}
    sessions = Path(args.sessions)
    sessions.mkdir(parents=True, exist_ok=True)

    # MLflow is mandatory: all experimentation is logged for full reproducibility
    # (owner directive). The engine step logs per-turn trajectory/prompt/output +
    # adapter-cond/prompt token metrics under this run (graph.py).
    pool_sha = hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()
    configure_mlflow(args.experiment)
    params = {
        **cfg.to_dict(),
        "arm": args.arm,
        "pool_path": str(args.tasks),
        "pool_sha256": pool_sha,
        "n_tasks": len(tasks),
    }
    with tracked_run(f"{args.arm}-seed{args.seed}", params=params):
        log_dataset(Path(args.tasks), name=Path(args.tasks).name, context="test")
        result = asyncio.run(
            run_benchmark(tasks, engine, config, sessions_dir=sessions)
        )
        mlflow.log_metric("pass_at_1", result.pass_at_1)
        mlflow.log_metric("passed_tasks", result.passed_tasks)
        mlflow.log_metric("total_tasks", result.total_tasks)

    out = {
        "arm": args.arm,
        "seed": args.seed,
        "adapter_scaling": cfg.adapter_scaling,
        "checkpoint": cfg.checkpoint_path,
        "pass_at_1": result.pass_at_1,
        "passed_tasks": result.passed_tasks,
        "total_tasks": result.total_tasks,
        "per_task": [
            {"task_id": r.task_id, "passed": r.passed} for r in result.per_task
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"{args.arm}: pass@1={result.pass_at_1:.3f} "
        f"({result.passed_tasks}/{result.total_tasks}) -> {args.out}"
    )


# ---------------------------------------------------------------------------- score
def _score_code(code: str, test_code: str) -> bool:
    from rune.engine.continuation import strip_self_tests  # noqa: PLC0415
    from rune.sandbox.executor import run_in_sandbox  # noqa: PLC0415

    full = strip_self_tests(code or "") + "\n\n" + test_code
    fb = run_in_sandbox(full)
    return bool(fb.exit_code == 0)


def cmd_score(args: argparse.Namespace) -> None:
    """Post-hoc per-turn scoring of one arm's sessions vs held-out tests.

    Emits, per task: the ordered list of code-producing turns with held-out
    pass/fail, attempt1/final, and per-step adapter-cond vs prompt token counts.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    from rune.config import load_rune_config  # noqa: PLC0415

    pool = {t["task_id"]: t for t in json.loads(Path(args.tasks).read_text())}
    tok = AutoTokenizer.from_pretrained(load_rune_config(None).model_id)
    sessions = Path(args.sessions)

    rows: list[dict[str, Any]] = []
    # task_ids contain a slash (e.g. "mbpp/106") -> nested session dirs, so walk
    # for session.jsonl and reconstruct the id from the path under `sessions`.
    for sess in sorted(sessions.rglob("session.jsonl")):
        tid = str(sess.parent.relative_to(sessions))
        if tid not in pool:
            continue
        test_code = pool[tid]["test_code"]
        steps = [
            json.loads(line)
            for line in sess.read_text().splitlines()
            if line.strip()
        ]
        turns: list[dict[str, Any]] = []
        for s in steps:
            adapter_toks = len(
                tok(s.get("trajectory", ""), add_special_tokens=False).input_ids
            )
            prompt_toks = len(
                tok(s.get("prompt", ""), add_special_tokens=False).input_ids
            )
            rec: dict[str, Any] = {
                "step": s.get("step"),
                "action": s.get("action"),
                "adapter_cond_tokens": adapter_toks,
                "prompt_tokens": prompt_toks,
            }
            if s.get("action") in _CODE_ACTIONS:
                rec["passed"] = _score_code(s.get("output", ""), test_code)
                rec["code"] = s.get("output", "")
            turns.append(rec)
        code_turns = [t for t in turns if t["action"] in _CODE_ACTIONS]
        rows.append(
            {
                "task_id": tid,
                "n_code_turns": len(code_turns),
                "attempt1_passed": code_turns[0]["passed"] if code_turns else None,
                "final_passed": code_turns[-1]["passed"] if code_turns else None,
                "code_turn_passes": [t["passed"] for t in code_turns],
                "turns": turns,
            }
        )
    out = {"arm": args.arm, "tasks": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    n = len(rows)
    a1 = sum(1 for r in rows if r["attempt1_passed"])
    fin = sum(1 for r in rows if r["final_passed"])
    print(
        f"{args.arm}: scored {n} tasks; attempt1 {a1}/{n}, "
        f"final {fin}/{n} -> {args.out}"
    )


# -------------------------------------------------------------------------- analyze
def _bootstrap_ci(
    deltas: list[float], n_boot: int = 10000
) -> tuple[float, float, float]:
    """Paired bootstrap mean + 95% CI over a list of per-task deltas.

    Deterministic resampling (index = (i*1103515245 + c) % n) — no RNG, so the
    analysis is reproducible across runs (Date/random are off-limits in workflows
    anyway and we keep the same discipline here).
    """
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(deltas) / n
    means: list[float] = []
    c = 12345
    for _b in range(n_boot):
        c = (c * 1103515245 + 12345) & 0x7FFFFFFF
        seed = c
        total = 0.0
        for _ in range(n):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            total += deltas[seed % n]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return mean, lo, hi


def cmd_analyze(args: argparse.Namespace) -> None:
    arms = {
        a: json.loads(Path(p).read_text())
        for a, p in (s.split("=", 1) for s in args.scored)
    }
    by_arm_task: dict[str, dict[str, dict[str, Any]]] = {
        a: {r["task_id"]: r for r in d["tasks"]} for a, d in arms.items()
    }
    # Slice: tasks where scale0 fails attempt-1 (arm-independent, pre-committed cut).
    scale0 = by_arm_task.get("scale0", {})
    slice_ids = sorted(
        tid for tid, r in scale0.items() if r["attempt1_passed"] is False
    )
    common = [
        tid for tid in slice_ids if all(tid in by_arm_task[a] for a in by_arm_task)
    ]

    report: dict[str, Any] = {
        "n_candidate": len(scale0),
        "slice_rule": "scale0 attempt-1 FAIL (held-out tests)",
        "slice_n": len(common),
        "slice_task_ids": common,
        "arms": {},
        "paired_vs_scale0": {},
    }
    for a in by_arm_task:
        rs = [by_arm_task[a][t] for t in common]
        a1 = sum(1 for r in rs if r["attempt1_passed"])
        fin = sum(1 for r in rs if r["final_passed"])
        # success-vs-turn curve: fraction with a PASS by code-turn k (cumulative).
        maxk = max((r["n_code_turns"] for r in rs), default=0)
        curve = []
        for k in range(1, maxk + 1):
            passed_by_k = sum(1 for r in rs if any(r["code_turn_passes"][:k]))
            curve.append(passed_by_k / len(rs) if rs else 0.0)
        report["arms"][a] = {
            "attempt1": a1,
            "final": fin,
            "n": len(rs),
            "recovery_gap": (fin - a1) / len(rs) if rs else 0.0,
            "cumulative_pass_by_turn": curve,
        }
    # paired deltas vs scale0 on final success + recovery gap
    for a in by_arm_task:
        if a == "scale0":
            continue
        fin_deltas, rec_deltas = [], []
        for t in common:
            ra, rs0 = by_arm_task[a][t], scale0[t]
            fin_deltas.append(
                int(bool(ra["final_passed"])) - int(bool(rs0["final_passed"]))
            )
            rg_a = int(bool(ra["final_passed"])) - int(bool(ra["attempt1_passed"]))
            rg_0 = int(bool(rs0["final_passed"])) - int(bool(rs0["attempt1_passed"]))
            rec_deltas.append(rg_a - rg_0)
        fm, flo, fhi = _bootstrap_ci(fin_deltas)
        rm, rlo, rhi = _bootstrap_ci(rec_deltas)
        report["paired_vs_scale0"][a] = {
            "final_success_delta": {"mean": fm, "ci": [flo, fhi]},
            "recovery_gap_delta": {"mean": rm, "ci": [rlo, rhi]},
        }
    Path(args.out).write_text(json.dumps(report, indent=2))
    keys = ("slice_n", "arms", "paired_vs_scale0")
    print(json.dumps({k: report[k] for k in keys}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parity")
    p.set_defaults(func=cmd_parity)

    p = sub.add_parser("run")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--tasks", required=True)
    p.add_argument("--sessions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=10, dest="max_iters")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--experiment", default="issue52-goal3-multiturn")
    p.add_argument("--prompt-mode", default="full", dest="prompt_mode")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score")
    p.add_argument("--arm", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--sessions", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("analyze")
    p.add_argument("--scored", nargs="+", required=True, help="arm=path.json ...")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
