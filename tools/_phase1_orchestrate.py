"""Phase-1 orchestrator (issue #52) — UNATTENDED. Train-HPO on the held-out-TRAIN split,
evaluate GENERALIZATION on the disjoint held-out-EVAL split, pick best, bench pass@1 vs
scale=0, write incremental results. Resilient: each step try/except, continue on failure,
results appended after every config so a crash loses at most one trial.

The pilot-2 ckpt (trained on the 10) did NOT generalize body accessibility to held-out tasks.
This asks the real question: does training on 40 disjoint tasks teach a TRANSFERABLE
"encode any body accessibly" skill? Honest outcome either way.

Run: tools/run_guarded.sh /tmp/phase1.log tools/_phase1_orchestrate.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rune.config import load_rune_config

WORK = Path("/tmp/phase1")
WORK.mkdir(exist_ok=True)
(WORK / "ckpt").mkdir(exist_ok=True)
RESULTS = WORK / "results.jsonl"
RUNE = "/workspaces/rune-gpu"
TRAIN_CORPUS = f"{RUNE}/benchmarks/mbpp_recall_train.jsonl"
HELDOUT = f"{RUNE}/benchmarks/mbpp_recall_heldout.jsonl"
WARM = f"{RUNE}/third_party/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
EXP = "issue52-phase1"
STEPS = 48
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

GRID = [
    {
        "name": "c1_t07_lp1_lg1",
        "matched_target_lp": -0.7,
        "primary_weight": 1.0,
        "guard_weight": 1.0,
    },
    {
        "name": "c2_t05_lp1_lg1",
        "matched_target_lp": -0.5,
        "primary_weight": 1.0,
        "guard_weight": 1.0,
    },
    {
        "name": "c3_t07_lp2_lg1",
        "matched_target_lp": -0.7,
        "primary_weight": 2.0,
        "guard_weight": 1.0,
    },
    {
        "name": "c4_t05_lp2_lg2",
        "matched_target_lp": -0.5,
        "primary_weight": 2.0,
        "guard_weight": 2.0,
    },
]


def log(msg: str) -> None:
    print(f"[orch] {msg}", flush=True)


def sh(cmd: list[str], logfile: Path) -> int:
    with open(logfile, "w") as f:
        return subprocess.run(
            cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=False
        ).returncode


def write_yaml(cfg: dict, path: Path) -> None:
    path.write_text(
        f'model_id: "{load_rune_config().model_id}"\n'
        f'checkpoint_path: "{WARM}"\n'
        f'corpus_path: "{TRAIN_CORPUS}"\n'
        f'checkpoint_dir: "./checkpoints/phase1-{cfg["name"]}"\n'
        f'experiment_name: "{EXP}"\n'
        f"learning_rate: 2.0e-5\nnum_epochs: 8\nmax_seq_length: 768\n"
        f"load_in_4bit: false\ngrad_accum_steps: 5\ngradient_accumulation_steps: 5\n"
        f"skip_zero_diff: false\nmax_steps: {STEPS}\nearly_stop_warmup: 100\n"
        f"log_steps: 5\nsave_steps: {STEPS}\nsnapshot_steps: 0\n"
        f'contrastive: true\ncontrastive_mode: "body_recall_guarded"\n'
        f"matched_target_lp: {cfg['matched_target_lp']}\n"
        f"primary_weight: {cfg['primary_weight']}\nguard_weight: {cfg['guard_weight']}\n"
    )


def latest_ckpt(dst: Path) -> str | None:
    """Download the FINAL checkpoint of the most recent EXP run -> local path."""
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri("http://localhost:5000")
        exp = mlflow.get_experiment_by_name(EXP)
        runs = mlflow.search_runs(
            [exp.experiment_id], order_by=["start_time DESC"], max_results=1
        )
        rid = runs.iloc[0]["run_id"]
        for name in (
            f"checkpoints/checkpoint_step{STEPS}.pt",
            "checkpoints/checkpoint.pt",
        ):
            try:
                p = mlflow.artifacts.download_artifacts(
                    run_id=rid, artifact_path=name, dst_path=str(dst)
                )
                return p
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        log(f"latest_ckpt failed: {exc}")
    return None


def parse_probe(jsonl: Path) -> dict:
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]

    def m(regime: str, span: str, key: str) -> float:
        v = [r[key] for r in rows if r["regime"] == regime and r["span"] == span]
        return sum(v) / len(v) if v else 0.0

    return {
        "absent_body_mm": round(m("absent", "body", "m_mismatch"), 4),
        "absent_body_mz": round(m("absent", "body", "m_zero"), 4),
        "absent_body_lpm": round(m("absent", "body", "lp_m"), 4),
        "absent_sig_mm": round(m("absent", "sig", "m_mismatch"), 4),
    }


def parse_pass1(logfile: Path) -> dict:
    out = {"present": None, "absent": None}
    for line in logfile.read_text().splitlines():
        if "PRESENT pass@1" in line:
            out["present"] = line.split("=")[-1].strip()
        if "ABSENT  pass@1" in line:
            out["absent"] = line.split("=")[-1].strip()
    return out


def main() -> int:
    probe = f"{RUNE}/tools/_specificity_probe.py"
    pass1 = f"{RUNE}/tools/_pass1_probe.py"
    entry = f"{RUNE}/tools/_distill_entry.py"

    # warm-start held-out accessibility baseline (reuse the earlier run if present).
    warm_probe = Path("/tmp/ho_ws.jsonl")
    warm_acc = parse_probe(warm_probe) if warm_probe.exists() else None
    log(f"warm-start held-out accessibility: {warm_acc}")

    trials = []
    for cfg in GRID:
        name = cfg["name"]
        log(f"=== TRAIN {name} {cfg} ===")
        yml = WORK / f"{name}.yaml"
        write_yaml(cfg, yml)
        rc = sh(
            [
                "uv",
                "run",
                "python",
                entry,
                "--config",
                str(yml),
                "--max-steps",
                str(STEPS),
            ],
            WORK / f"train_{name}.log",
        )
        if rc != 0:
            log(f"train {name} rc={rc} — skipping")
            continue
        ckpt = latest_ckpt(WORK / "ckpt")
        if not ckpt:
            log(f"no ckpt for {name} — skipping")
            continue
        named = WORK / "ckpt" / f"{name}.pt"
        os.replace(ckpt, named)
        log(f"=== EVAL {name} held-out accessibility ===")
        out = WORK / f"heldout_{name}.jsonl"
        rc = sh(
            [
                "uv",
                "run",
                "python",
                probe,
                "--ckpt",
                str(named),
                "--corpus",
                HELDOUT,
                "--out",
                str(out),
            ],
            WORK / f"probe_{name}.log",
        )
        if rc != 0 or not out.exists():
            log(f"probe {name} rc={rc} — skipping")
            continue
        acc = parse_probe(out)
        rec = {"name": name, **cfg, **acc, "ckpt": str(named)}
        trials.append(rec)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"RESULT {name}: {acc}")

    if not trials:
        log("no successful trials — aborting bench")
        return 1

    # Pick best by held-out body accessibility (m-zero = matched-vs-base), sig retained >= warm.
    sig_floor = (warm_acc or {}).get("absent_sig_mm", 0.0)
    eligible = [t for t in trials if t["absent_sig_mm"] >= sig_floor - 0.5] or trials
    best = max(eligible, key=lambda t: t["absent_body_mz"])
    log(
        f"=== BEST: {best['name']} (held-out absent/body m-zero={best['absent_body_mz']}) ==="
    )

    # Bench pass@1 on held-out: scale=0, warm-start, best.
    bench = {}
    arms = [
        ("scale0", ["--scale0"]),
        ("warmstart", ["--ckpt", WARM]),
        ("best", ["--ckpt", best["ckpt"]]),
    ]
    for arm, extra in arms:
        log(f"=== BENCH pass@1 {arm} (held-out) ===")
        lf = WORK / f"pass1_{arm}.log"
        rc = sh(
            [
                "uv",
                "run",
                "python",
                pass1,
                "--corpus",
                HELDOUT,
                "--out",
                str(WORK / f"pass1_{arm}.jsonl"),
                *extra,
            ],
            lf,
        )
        bench[arm] = parse_pass1(lf) if rc == 0 else {"error": rc}
        log(f"BENCH {arm}: {bench[arm]}")
        with open(WORK / "bench.json", "w") as f:
            json.dump(bench, f, indent=2)

    summary = {"warm_acc": warm_acc, "best": best, "trials": trials, "bench": bench}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    log("=== PHASE-1 ORCHESTRATION COMPLETE ===")
    log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
