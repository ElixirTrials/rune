"""Mining yield probe: generate engine sessions whose diagnose->repair fires on a
SEMANTIC signal, then measure fail->repair->pass yield (advisor option (ii)).

The product engine executes ``run_in_sandbox(strip_self_tests(code))`` mid-loop
with NO functional test -> failures are only syntax/import/load-time, so a harvest
would be pure syntax-repair (vacuous for #52's "what we tried & why it failed").
This MINING-ONLY harness monkeypatches ``rune.engine.graph.run_in_sandbox`` to
append each task's VISIBLE example assert (from the description `>>> assert ...`,
NOT the held-out test_code = no leakage) before executing, so wrong-but-runnable
code FAILS the assert -> the engine's existing diagnose->repair loop fires on a
semantic error. No src/rune change; the product engine is untouched.

Yield-first: run the 10 frozen tasks once with sessions_dir; report fail->repair->
pass chains, whether diagnoses are semantic, and VALID avoid episodes (same target;
failed output present; passing repair present; critique non-empty; repair changes
the failed region). Build the extractor/utility apparatus ONLY if yield > 0.

Run in RUNE's venv:
  uv run python tools/_mine_semantic_sessions.py --out /tmp/mine_sem_sessions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

RUNE = "/workspaces/rune-gpu"
TASKS_FILE = f"{RUNE}/benchmarks/mbpp_phase0_iter.json"
CKPT = (
    f"{RUNE}/third_party/doc-to-lora/trained_d2l/"
    "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
BASE = "Qwen/Qwen3-4B-Instruct-2507"
_ASSERT_RE = re.compile(r">>>\s*(assert .+)")

# Per-task example assert, set before each engine.ainvoke (run loop is sequential,
# so a module global is race-free across tasks; MBPP _main has no sibling subtasks).
_CURRENT_ASSERT = ""


def _example_assert(description: str) -> str:
    m = _ASSERT_RE.search(description)
    return m.group(1).strip() if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", type=Path, default=Path(TASKS_FILE))
    ap.add_argument("--out", type=Path, default=Path("/tmp/mine_sem_sessions"))
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--checkpoint-path", type=str, default=CKPT)
    ap.add_argument("--adapter-scaling", type=float, default=1.0)
    a = ap.parse_args()

    import rune.engine.graph as graph  # noqa: PLC0415
    from rune.config import PipelineConfig  # noqa: PLC0415
    from rune.engine.graph import create_engine  # noqa: PLC0415
    from rune.engine.state import make_initial_state  # noqa: PLC0415
    from rune.mining.session_log import write_session  # noqa: PLC0415
    from rune.model.wrapper import ModelWrapper  # noqa: PLC0415

    real_run = graph.run_in_sandbox

    def patched_run(code, **kw):
        # Append the current task's VISIBLE example assert -> semantic signal.
        augmented = code + "\n" + _CURRENT_ASSERT + "\n" if _CURRENT_ASSERT else code
        return real_run(augmented, **kw)

    graph.run_in_sandbox = patched_run  # mining-only monkeypatch

    cfg = PipelineConfig().override(
        model_id=a.model_id,
        checkpoint_path=a.checkpoint_path,
        adapter_scaling=a.adapter_scaling,
    )
    tasks = json.loads(a.tasks_file.read_text())
    model = ModelWrapper.from_config(cfg)
    engine = create_engine()
    budget = cfg.to_dict()["max_phase_iterations"]
    run_config = cfg.to_dict()
    a.out.mkdir(parents=True, exist_ok=True)

    global _CURRENT_ASSERT
    for t in tasks:
        _CURRENT_ASSERT = _example_assert(t["description"])
        print(f"\n=== {t['task_id']} assert={_CURRENT_ASSERT[:60]!r} ===", flush=True)
        state = make_initial_state(t["description"], budget)
        try:
            final = asyncio.run(
                engine.ainvoke(
                    state,
                    config={"configurable": {"model": model, "run_config": run_config}},
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  engine error: {exc}", flush=True)
            continue
        write_session(
            final,
            {"benchmark": "mbpp_sem", "problem_id": t["task_id"].split("/")[-1]},
            a.out / t["task_id"].replace("/", "_"),
        )
        # quick per-task chain readout from the trajectory
        traj = final.get("trajectory", [])
        per_target: dict[str, list[tuple[int, str, int]]] = {}
        for rec in traj:
            ec = rec.feedback.exit_code if rec.feedback is not None else None
            per_target.setdefault(rec.target_subtask or "", []).append(
                (rec.step, rec.action_name, ec if ec is not None else 999)
            )
        for tgt, seq in per_target.items():
            codes = [(s, act, ec) for s, act, ec in seq]
            print(f"  target={tgt}: {codes}", flush=True)

    # ---- yield analysis over written sessions ----
    print("\n=== YIELD ANALYSIS ===", flush=True)
    from rune.mining.miner import load_session  # noqa: PLC0415

    n_sessions = n_chains = n_valid = 0
    for sd in sorted(a.out.glob("*")):
        if not (sd / "session.jsonl").exists():
            continue
        n_sessions += 1
        steps, _meta = load_session(sd)
        by_t: dict[str, list[dict]] = {}
        for s in steps:
            by_t.setdefault(s.get("target") or "", []).append(s)
        for tgt, seq in by_t.items():
            seq.sort(key=lambda s: s.get("step", 0))
            # find a failed step (exit_code!=0) followed by a later passing step
            for i, s in enumerate(seq):
                fb = s.get("feedback") or {}
                if fb.get("exit_code") not in (None, 0):  # failed (incl -1 timeout)
                    later_pass = next(
                        (
                            q
                            for q in seq[i + 1 :]
                            if (q.get("feedback") or {}).get("exit_code") == 0
                        ),
                        None,
                    )
                    if later_pass is None:
                        continue
                    n_chains += 1
                    failed_out = s.get("output", "")
                    fixed_out = later_pass.get("output", "")
                    critique = later_pass.get("trajectory", "")
                    has_crit = "## Review Feedback" in critique and bool(
                        critique.split("## Review Feedback", 1)[-1].strip()
                    )
                    changed = failed_out.strip() != fixed_out.strip()
                    valid = bool(failed_out) and bool(fixed_out) and has_crit and changed
                    n_valid += int(valid)
                    print(
                        f"  CHAIN {sd.name}/{tgt}: fail@step{s.get('step')}"
                        f"(ec={fb.get('exit_code')}) -> pass@step{later_pass.get('step')}"
                        f" | valid={valid} (crit={has_crit} changed={changed})",
                        flush=True,
                    )
                    break
    print(
        f"\nSESSIONS={n_sessions}  FAIL->PASS chains={n_chains}  VALID episodes={n_valid}",
        flush=True,
    )
    print(
        "VERDICT: "
        + (
            "nonzero valid yield -> build extractor/apparatus"
            if n_valid > 0
            else "SPARSE -> do NOT scale to 257; pre-registered (iii-a harder tasks / iii-b temp-sampled pairs)"
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
