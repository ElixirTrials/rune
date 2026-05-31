# Collapse Metrics

Pure metric helpers used to detect inert or content-independent generated
adapters during HyperLoRA training.

The implementation is landing on the Issue #49 training branch. This page is
kept narrative-only in the docs PR so it does not require branch-only Python
modules to exist on `main`.

The metric family covers:

- optimizer coverage for collapse-critical parameters,
- teacher/base diff-token fraction,
- student agreement with teacher on diff tokens,
- preservation on teacher/base agreement tokens,
- gradient and parameter summaries for `scaler_B`, heads, and aggregators.
