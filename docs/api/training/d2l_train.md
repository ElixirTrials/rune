# Distillation Training

Declares the D2LTrainConfig pydantic base config and the run_distillation entrypoint that loads the JSONL corpus and runs a DiffAwareSFTTrainer (built via trl.SFTConfig + LoRA) for round-2 distillation.

::: rune.training.d2l_train
