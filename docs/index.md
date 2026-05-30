# Rune

Local-first coding agent that encodes coding trajectories into LoRA adapters via a perceiver hypernetwork (ctx-to-lora / HyperLoRA).

## Overview

Rune exposes four concerns as CLI commands:

- **Run** — execute a coding task through the engine (`rune run`)
- **Train** — oracle QLoRA → hypernetwork distillation → success gate (`rune train`)
- **Mine** — extract per-action coding trajectories from sessions (`rune mine`)
- **Bench** — pass@1 benchmarking with optional Optuna HPO (`rune bench`)

## Quick Start

```bash
uv sync
uv run rune run "build a calculator"
```

## Architecture

The engine is a single LangGraph `step` node with a conditional self-edge (`should_continue`) that loops until the budget is exhausted or no actions remain. Each iteration, the policy selects actions, a hypernetwork generates a task-specific LoRA adapter from the trajectory text, the adapter is hot-swapped onto the base model, and the output is generated and sandbox-executed.

The action sequence is `decompose → plan → code → [diagnose → repair]* → integrate`:

1. **Decompose** the task into subtasks forming a dependency DAG.
2. **Plan** each subtask, layer by layer.
3. **Code** each subtask, executing the result in the sandbox.
4. **Diagnose → repair** — a bounded sub-loop on failing subtasks (`MAX_REPAIRS=2`, `MAX_RETRIES=4`).
5. **Integrate** — terminal, gated on all subtasks passing.

Independent subtasks in the same DAG layer are dispatched and sandbox-executed in parallel within a single step. A complexity gate skips decomposition for short single-unit tasks, running them against a synthetic `_main` subtask.

See [engine](architecture/engine.md), [model](architecture/model.md), and [training](architecture/training.md) for details.
