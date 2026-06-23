# Rune

Rune is a local-first coding-agent research project that encodes coding
trajectories into generated LoRA adapters via a ctx-to-lora / HyperLoRA
perceiver hypernetwork.

The core hypothesis is **adapter-as-memory**: an agent should be able to carry
the useful parts of its coding history in LoRA weights instead of continually
growing the prompt. If this works, a local model can iterate on long tasks with
a constant visible context while the adapter carries compressed trajectory
state.

## Overview

Rune exposes four concerns as CLI commands:

- **Mine** — extract coding trajectories from sessions (`rune mine`)
- **Train** — recover and validate HyperLoRA adapters (`rune train`)
- **Run** — execute a coding task through the engine (`rune run`)
- **Bench** — measure pass@1 and run HPO (`rune bench`)

An auxiliary `rune gen-tasks` command materializes benchmark task files for the runner.

The north-star metric is pass@1 on coding benchmarks, comparing
adapter-conditioned runs against the frozen base model. Leading diagnostics also
track degeneration, syntax validity, content retrieval, preservation, and
teacher-vs-base diff-token signal.

## Quick Start

```bash
uv sync
uv run pytest tests/unit/ -q
uv run rune --help
```

GPU training and benchmark runs should follow the memory rules in `CLAUDE.md`:
check memory before model load, keep GPU imports deferred, and run long jobs
under the RAM watchdog.

## Architecture

The engine is a single LangGraph `step` node with a conditional self-edge
(`should_continue`) that loops until the budget is exhausted or no actions
remain. Each iteration, the policy selects actions, the hypernetwork generates a
task-specific LoRA adapter from the trajectory text, the adapter is hot-swapped
onto the base model, and the output is generated and sandbox-executed.

The action sequence is `decompose → plan → code → [diagnose → repair]* → integrate`:

1. **Decompose** the task into subtasks forming a dependency DAG.
2. **Plan** each subtask, layer by layer.
3. **Code** each subtask, executing the result in the sandbox.
4. **Diagnose → repair** — a bounded sub-loop on failing subtasks (`MAX_REPAIRS=4`, `MAX_RETRIES=8`).
5. **Integrate** — terminal, gated on all subtasks passing.

Independent subtasks in the same DAG layer are dispatched and sandbox-executed in parallel within a single step. `decompose` always runs first; the model itself decides whether a self-contained task is one subtask or many (no word-count heuristic pre-empts it), and a lone passing subtask finalizes directly without a separate `integrate` step.

See [engine](architecture/engine.md), [model](architecture/model.md), and [training](architecture/training.md) for details.

## Active Research Thread

Current work centers on D2L-style privileged-context self-distillation:

1. The teacher is the frozen base model with trajectory context in prompt.
2. The student is the same base model plus a generated adapter, with trajectory
   context removed.
3. The training objective matches teacher top-K logits on answer-span tokens
   where the teacher and base differ.

This is designed to force the generated adapter to internalize what the teacher
got from context. Promotion requires content and benchmark gates, not adapter
magnitude.
