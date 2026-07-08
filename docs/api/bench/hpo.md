# HPO

Runs an Optuna study that tunes engine parameters (adapter_scaling, temperature, presence_penalty, max_phase_iterations, cont_multiplier, plus an optional categorical prompt_mode) over the benchmark suite to maximise pass@1, logging trials to MLflow.

::: rune.bench.hpo
