# Retired Oracle Cache

The previous oracle-adapter cache was removed from the active training path.
Rune currently trains the HyperLoRA generator with privileged-context D2L
self-distillation: the teacher is the frozen base model with trajectory context
in prompt, not a per-bin oracle adapter.

This page is retained only to explain the API removal for older links.
