"""Print Optuna HPO progress (REMOVE-BEFORE-MERGE).

One line for a periodic overnight Monitor / PR update: completed-trial count,
running best tuning pass@1, and the best params so far. Reads the persisted
sqlite study so it works while the HPO is mid-run (and after, for recovery).
"""

from __future__ import annotations

import sys

import optuna

DB = "optuna_bench_hpo.db"
STUDY = "rune-bench-hpo"


def main() -> None:
    try:
        study = optuna.load_study(study_name=STUDY, storage=f"sqlite:///{DB}")
    except Exception as e:  # noqa: BLE001
        print(f"HPO: no study yet ({e})")
        return
    trials = study.get_trials(deepcopy=False)
    done = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    if not done:
        print(f"HPO: 0 complete, {len(running)} running")
        return
    best = study.best_trial
    bp = {
        k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.params.items()
    }
    print(
        f"HPO: {len(done)} complete / {len(running)} running | "
        f"best tuning_pass@1={best.value:.3f} @ {bp}"
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
