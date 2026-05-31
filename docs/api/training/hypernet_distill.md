# HyperLoRA Distillation

Active D2L privileged-context self-distillation loop for the HyperLoRA
hypernetwork. This is the current Issue #49 training path.

The implementation is landing on the Issue #49 training branch. This page is
kept narrative-only in the docs PR so the public documentation can describe the
active research path without importing branch-only Python modules from `main`.

At a high level:

- teacher = frozen base model with trajectory context,
- student = frozen base model plus generated adapter with trajectory removed,
- objective = top-K KL on answer-span tokens where teacher and base disagree,
- guardrails = diff-token movement, preservation, skip fraction, and
  degeneration checks.
