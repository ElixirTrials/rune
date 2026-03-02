# Risk Matrix

This appendix enumerates the primary research risks identified during architecture research. Each risk is assigned to the implementation phase where it is most relevant and where its mitigation is tested. Risks are ordered by severity.

| Risk | Phase | Severity | Mitigation | Warning Signs |
|------|-------|----------|------------|---------------|
| **Hypernetwork mode collapse** | Phase 1 (kill-switch gate), Phase 4 | High | Diversity regularization in training loss from the start. Monitor adapter cosine diversity metric (threshold > 0.1). The degenerate solution — the mean adapter — has near-zero variance across different inputs. | Generated adapters cluster tightly in PCA; cosine similarity between adapters approaches 1.0; training loss converges but Pass@1 does not improve. |
| **Adapter composition interference** | Phase 2 | Medium | Default to single-adapter retrieval at inference time. Treat multi-adapter composition as an optional future experiment. Direct additive merging of heterogeneous LoRAs produces interference in non-orthogonal subspaces. | Performance with 2 adapters loaded is worse than with 1; inference latency increases non-linearly with adapter count. |
| **Catastrophic forgetting** | Phase 2 | Medium | Immutable adapter storage with write-once semantics enforced at the registry API level. No code path may overwrite an existing adapter. Adapters indexed by session ID and timestamp. | Performance on session-1 tasks degrades after session-50 adapters are added to the corpus; adapter file checksums change unexpectedly. |

## How to Use This Matrix

Each risk listed here should be monitored during the assigned phase via MLflow experiment tracking. Log the warning sign metrics as MLflow scalars at each evaluation step so degradation is visible before it becomes unrecoverable. If any warning sign is observed, stop training and reassess before continuing — there are no predefined fallback strategies, only stop-and-learn.
