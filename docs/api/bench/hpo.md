# HPO

Runs an Optuna study that tunes engine parameters (adapter_scaling, temperature, max_tokens, max_phase_iterations, cont_multiplier) over the benchmark suite to maximise pass@1, logging trials to MLflow.

::: rune.bench.hpo
