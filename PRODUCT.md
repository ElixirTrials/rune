# PRODUCT.md

> **Read me first.** This file grounds every non-trivial technical decision Claude makes
> in this repo. Without it, defaults skew toward over-engineering.

## 1. North-star metric

**pass@1 on coding benchmarks** — fraction of benchmark tasks (HumanEval, MBPP, SWE-bench) solved correctly on the first attempt, comparing adapter-conditioned runs against the base model.

## 2. Users & personas

- **Primary**: ML/AI researchers exploring hypernetwork-based adapter generation for code. They care about reproducible benchmarks, fast iteration on training configs, and clear evidence that adapters improve over the base model.
- **Secondary**: The author (solo maintainer). Iterates on the engine loop, hypernetwork architecture, and training pipeline. Success = publishable results demonstrating adapter-as-memory.

## 3. Jobs to be done

1. **When** I have a coding task and limited hardware, **I want to** run it through a local agent with dynamically generated adapters, **so I can** get task completion without depending on API-based models or unbounded context windows.
2. **When** working on a multi-step coding task, **I want to** have the agent carry learned context via trajectory-conditioned adapters (adapter-as-memory), **so I can** get improved continuation quality across rounds without growing the prompt.
3. **When** I need to continue generation beyond the context limit, **I want to** use adapters as a constant-length substrate for continuation, **so I can** produce arbitrarily long outputs on hardware that can't fit the full context in the prompt.

## 4. Regulatory surface

Not applicable. Rune is a standalone local-first research tool. No PHI, no PII beyond the user's own code, no cloud service, no patient-facing decisions.

## 5. Do-not-break invariants

1. **Benchmark reproducibility** — same config YAML + same checkpoint = same pass@1 within statistical noise. How it fails loudly: benchmark runner logs per-task pass/fail and aggregate score; any config or seeding change that shifts results is visible in MLflow.
2. **CPU-importable** — all GPU imports (torch, transformers, peft, trl, flash-attn) are deferred inside function bodies. CI runs on CPU-only. How it fails loudly: `uv run pytest tests/unit/ -q` and `uv run mypy src/` fail on import if this regresses.
3. **Safe adapter hot-swap** — `hotswap_adapter` replaces LoRA weights in-place via `set_peft_model_state_dict` without corrupting base model weights. How it fails loudly: any base-weight mutation would cause benchmark scores to drift across runs within a session.

## 6. Out-of-scope (explicit non-goals)

- **Cloud/API service** — rune stays local-first. No hosted inference endpoint.
- **IDE integration** — no VS Code extension or editor plugin. CLI only.
- **Production coding assistant** — this is a research tool for proving adapter-as-memory, not a product to ship to end users.

## 7. Success metrics & current bets

- **Leading metrics** (measured per experiment run): pass@1, adapter-conditioned pass@1 lift over base, degeneration rate in continuation rounds, syntax-valid rate of generated code.
- **Current bet**: Hypernetwork-generated LoRA adapters both carry task memory across continuation rounds AND provide unbounded context at constant prompt length, beating the base model's pass@1 on coding benchmarks.
- **Kill criteria**: If the best HPO-tuned adapter configuration does not beat base-model pass@1 on HumanEval/MBPP, the adapter-as-memory approach is not viable.

## 8. Open product questions

1. **Which benchmarks beyond HumanEval/MBPP?** SWE-bench? APPS? Custom domain-specific tasks? Benchmark selection affects what "winning" means.
2. **Multi-language support?** Currently Python-only (generation, sandbox, tree-sitter parsing). If the approach works, extending to other languages is a separate decision.

## 9. Glossary

- **Adapter-as-memory** — using trajectory-conditioned LoRA weights to carry task context across continuation rounds, replacing prompt-based context accumulation.
- **Continuation round** — a follow-up generation pass where the model extends partially-generated code, using a scaled adapter (`cont_multiplier` over base `adapter_scaling`).
- **HyperLoRA** — the perceiver hypernetwork (from ctx-to-lora) that takes trajectory activations and produces per-layer LoRA A/B weights.
- **Trajectory** — serialized record of a coding session's actions, inputs, outputs, and feedback, used as conditioning input to the hypernetwork.
- **Success gate** — post-training evaluation that compares new benchmark scores against a baseline; must show improvement on ≥4 benchmarks with no regressions.
- **Hot-swap** — replacing the active LoRA adapter weights in a PEFT model in-place without reloading the base model.
- **Degeneration score** — 4-gram repetition ratio used to detect when continuation output is degenerating into repetitive patterns (threshold: 0.5).
- **pass@1** — fraction of benchmark problems solved correctly on the first attempt (no retries).
