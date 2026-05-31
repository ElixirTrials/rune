# Orchestrator

Dispatches the active HyperLoRA D2L distillation stage and optional Optuna HPO
wrapper. The benchmark success gate is evaluated outside the inline training
pipeline, after adapter checkpoints are trained and held-out evaluation is
wired.

::: rune.training.orchestrator
