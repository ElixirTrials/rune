# D2L Train Config

Declares the `D2LTrainConfig` pydantic base config shared by the active
HyperLoRA distillation loop. The GPU-heavy training implementation is in
`rune.training.hypernet_distill`; this module intentionally stays lightweight
and CPU-importable.

::: rune.training.d2l_train
