# Rune

Local-first coding agent that encodes coding trajectories into LoRA adapters via a perceiver hypernetwork.

## Overview

Rune is a single-loop LangGraph engine with four concerns:

- **Run** — Execute a coding task through the engine
- **Train** — Oracle → hypernetwork distillation → success gate
- **Mine** — Extract coding trajectories from sessions
- **Bench** — Benchmark with optional HPO

## Quick Start

```bash
uv sync
uv run rune run "build a calculator"
```

## Architecture

The engine follows a single recursive loop:

1. **Decompose** the task into subtasks with DAG dependencies
2. **Plan** each subtask (respecting dependency layers)
3. **Code** each subtask (with sandbox execution)
4. **Integrate** all passing code
5. **Diagnose** failures and retry

At each step, a hypernetwork generates a task-specific LoRA adapter from the coding trajectory.
