# Rune Documentation

Rune is a local-first coding agent that uses LoRA weight space as episodic memory. Rather than relying on context window size to retain knowledge across sessions, Rune generates task-specific LoRA adapters and retrieves them at inference time — building a persistent parametric memory that scales independently of context length. This site contains the implementation plan and supporting documentation for the project.

## Documentation

- [Implementation Plan](implementation-plan.md) — Phased build plan from hardware validation through hypernetwork training
- [Risk Matrix](appendices/risk-matrix.md) — Primary research risks with mitigations and warning signs
- [Build Order](appendices/build-order.md) — Component dependency chain and recommended build sequence
