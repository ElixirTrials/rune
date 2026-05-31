# Retired Training Config

The previous `rune.training.config` module and `Round2TrainConfig` oracle
routing surface have been removed from the active Issue #49 training path.

Use `rune.training.d2l_train.D2LTrainConfig` for shared config fields and
`rune.training.hypernet_distill.DistillConfig` for the active HyperLoRA D2L
distillation knobs.

This page is retained only to explain the API removal for older links.
