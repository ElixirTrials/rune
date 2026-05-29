# Orchestrator

Drives the three-stage training pipeline (oracle QLoRA stub -> hypernetwork distillation -> success gate), with an optional Optuna HPO mode over training hyperparameters; the success-gate scores are currently placeholders.

::: rune.training.orchestrator
