# Rune

Rune is a local-first coding-agent research project for testing a simple but
ambitious hypothesis: a coding agent can carry its working history in generated
LoRA weights instead of an ever-growing prompt.

The system serializes a coding trajectory, extracts base-model activations from
that trajectory, and uses a ctx-to-lora / HyperLoRA perceiver hypernetwork to
generate per-layer LoRA adapters. Those adapters are hot-swapped onto a frozen
base model during the agent loop, giving the model a constant-length
``adapter-as-memory`` signal while the visible prompt stays small.

## Research Goal

The north-star metric is pass@1 on coding benchmarks. Rune is useful only if
adapter-conditioned runs beat the base model on first-attempt task completion,
while preserving syntax validity and avoiding degeneration.

The current research bet is broader than benchmark lift alone:

- **Adapter-as-memory:** trajectory-conditioned adapters should improve
  multi-step coding behavior without appending all history to the prompt.
- **Constant-length continuation:** continuation rounds should carry prior work
  through weights rather than through unbounded context.
- **Local-first execution:** mining, training, running, and benchmarking should
  remain CLI-driven on local hardware.

Rune is not a hosted coding assistant or IDE plugin. It is a research tool for
proving whether hypernetwork-generated adapters can become useful working
memory for code.

## Current Shape

The package exposes four concerns:

- `rune mine` extracts coding trajectories for training.
- `rune train` runs HyperLoRA training and gates.
- `rune run` executes the single-loop LangGraph coding engine.
- `rune bench` measures pass@1 and supports Optuna sweeps.

The active training work is D2L-style privileged-context self-distillation:

- **Teacher:** frozen base model with trajectory context in prompt.
- **Student:** frozen base model plus generated adapter, with trajectory
  removed from prompt.
- **Loss:** top-K KL over the answer span, masked to positions where the
  in-context teacher differs from the base model.

This is a deliberately measured research path. Promotion depends on content
gates, dual-precision evaluation, and tiny coding benchmarks, not on adapter
magnitude alone.

## Quick Start

```bash
uv sync
uv run pytest tests/unit/ -q
uv run rune --help
```

GPU work should use the repository guardrails in `CLAUDE.md`: keep GPU imports
deferred, check memory before loading the base model, and run long jobs under
the RAM watchdog.

## Documentation

Local docs are built with MkDocs:

```bash
uv run mkdocs build
```

Start with:

- `PRODUCT.md` for product constraints and success criteria.
- `docs/index.md` for the research overview.
- `docs/architecture/` for the engine, model, and training architecture.
- `CLAUDE.md` for operational rules on this GPU workspace.

## Status

Rune is alpha research software. The engine, model wrapper, mining path,
training harnesses, and benchmark tooling are under active development. The
current priority is recovering a non-collapsed HyperLoRA training path and
showing adapter-conditioned pass@1 lift on real coding tasks.

## License

See `LICENSE`.
