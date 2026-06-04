"""Fetch goal-2 scaling checkpoints from MLflow (S3) to local staging paths. REMOVE-BEFORE-MERGE.

Distillation uploads checkpoints to MLflow and deletes the local copy. This maps each
`issue52-goal2-scaling` run to its corpus size (via the corpus_path param) and downloads
checkpoint_step48.pt -> /tmp/goal2/ckpt/c3_n{N}.pt.

Run: uv run python tools/_fetch_goal2_ckpts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXP = "issue52-goal2-scaling"
DST = Path("/tmp/goal2/ckpt")


def main() -> int:
    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri("http://localhost:5000")
    exp = mlflow.get_experiment_by_name(EXP)
    if exp is None:
        print(f"no experiment {EXP}", flush=True)
        return 1
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=20)
    seen: set[str] = set()
    for _, row in runs.iterrows():
        rid = row["run_id"]
        if row.get("status") != "FINISHED":
            continue  # skip RUNNING/active runs — their checkpoint isn't uploaded yet
        # corpus_path param tells us the size; find mbpp_recall_train_{N} or the base 40.
        corpus = None
        for col in row.index:
            if col.startswith("params.") and isinstance(row[col], str) and "mbpp_recall_train" in row[col]:
                corpus = row[col]
                break
        if corpus is None:
            continue
        m = re.search(r"mbpp_recall_train_(\d+)", corpus)
        n = m.group(1) if m else "40"
        if n in seen:
            continue
        for name in ("checkpoint_step48.pt", "checkpoint.pt", "checkpoint_best.pt"):
            try:
                p = mlflow.artifacts.download_artifacts(
                    run_id=rid, artifact_path=f"checkpoints/{name}", dst_path=str(DST / f"_dl_{n}")
                )
                out = DST / f"c3_n{n}.pt"
                Path(p).replace(out)
                print(f"[OK] N={n} run={rid[:8]} {name} -> {out} ({out.stat().st_size} B)", flush=True)
                seen.add(n)
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            print(f"[MISS] N={n} run={rid[:8]} no checkpoint artifact: {last}", flush=True)
    print(f"fetched sizes: {sorted(seen)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
