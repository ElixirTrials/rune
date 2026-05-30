# Diff-Aware Loss

Implements line-level diff-aware SFT loss weighting: a DiffWeightedDataCollator that aligns assistant spans to hunk char-ranges to up-weight changed tokens, plus DiffAwareSFTTrainer applying the weighted cross-entropy and per-step diagnostic metrics.

::: rune.training.diff_loss
